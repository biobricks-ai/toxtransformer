# cmd:
# PYTHONPATH=./ spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///data/tmp/spark-events --conf spark.local.dir=/data/tmp/spark-local code/2_3_build_tensordataset_from_augmented.py

import uuid, torch, torch.nn.utils.rnn
import pyspark.sql, pyspark.sql.functions as F
import cvae.utils, cvae.tokenizer.selfies_tokenizer, cvae.tokenizer.selfies_property_val_tokenizer
import logging
import pathlib
import pandas as pd
import shutil
import selfies as sf
from pyspark.sql import Window
from pyspark.sql.types import StringType, ArrayType, IntegerType
from pyspark.sql.functions import pandas_udf, udf, explode, array, col

logpath = pathlib.Path('cache/build_tensordataset/log')
logpath.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=logpath / 'build_tensordataset_augmented_5fold.log', level=logging.INFO, filemode='w')

spark = cvae.utils.get_spark_session()

# No longer need encoding functions since we have pre-encoded alternatives

# =============================================================================
# LOAD DATA WITH PRE-COMPUTED ALTERNATIVES
# =============================================================================

# Load data with alternative encoded SELFIES already included
data = spark.read.parquet("cache/preprocess_activities/activities_augmented.parquet").orderBy(F.rand(seed=42)) \
    .filter(F.col('source') != 'ctdbase')

logging.info("Initial data loading complete")
logging.info(f"Data includes 'alternative_encoded_selfies' column: {('alternative_encoded_selfies' in data.columns)}")

# Select and prepare base data
data = data.select('smiles', 'alternative_encoded_selfies', 'encoded_selfies', 'assay_index', 'value') \
    .groupby('smiles', 'alternative_encoded_selfies', 'encoded_selfies', 'assay_index') \
    .agg(F.collect_set('value').alias('values')) \
    .filter(F.size('values') == 1) \
    .select('smiles', 'alternative_encoded_selfies', 'encoded_selfies', 'assay_index', 
            F.element_at('values', 1).alias('value'))

# =============================================================================
# SPLIT DATA AND APPLY AUGMENTATION
# =============================================================================

# Analyze data distribution per assay-value combination
assay_value_counts = data.groupBy("assay_index", "value").count().withColumnRenamed("count", "total_count")

# Log initial distribution
logging.info("Initial assay-value distribution:")
initial_dist = assay_value_counts.collect()
for row in sorted(initial_dist, key=lambda x: (x['assay_index'], x['value'])):
    logging.info(f"Assay {row['assay_index']}, Value {row['value']}: {row['total_count']} examples")

# Filter to only include assay-value combinations that have at least 30 examples
MIN_EXAMPLES_PER_CLASS = 30
valid_assay_values = assay_value_counts.filter(F.col("total_count") >= MIN_EXAMPLES_PER_CLASS)

# Join back to keep only data points from valid assay-value combinations
data_filtered = data.join(valid_assay_values, on=["assay_index", "value"])

# Create window for stratified sampling
window = Window.partitionBy('assay_index', 'value').orderBy(F.rand(seed=42))
data_with_row_nums = data_filtered.withColumn("row_number", F.row_number().over(window))

# Assign 5-fold cross-validation splits
N_FOLDS = 5

@pandas_udf(IntegerType())
def assign_fold_udf(row_number: pd.Series, total_count: pd.Series) -> pd.Series:
    """Assign each sample to a fold (0-4) for 5-fold cross-validation."""
    result = pd.Series([0] * len(row_number))
    
    for i in range(len(row_number)):
        total = total_count.iloc[i]
        row_num = row_number.iloc[i]
        
        # Ensure each fold gets approximately 20% of the data
        fold_size = total // N_FOLDS
        remainder = total % N_FOLDS
        
        # Distribute remainder across first few folds
        cumulative = 0
        for fold in range(N_FOLDS):
            fold_count = fold_size + (1 if fold < remainder else 0)
            cumulative += fold_count
            if row_num <= cumulative:
                result.iloc[i] = fold
                break
    
    return result

data_with_folds = data_with_row_nums.withColumn("fold", 
    assign_fold_udf("row_number", "total_count")).cache()

logging.info("5-fold assignment complete")

# Log fold distribution
fold_counts = data_with_folds.groupBy("fold").count().orderBy("fold").collect()
for row in fold_counts:
    logging.info(f"Fold {row['fold']}: {row['count']} examples")

# Verify stratification
fold_assay_dist = data_with_folds.groupBy("fold", "assay_index", "value").count() \
    .orderBy("fold", "assay_index", "value").collect()
logging.info("\nFold stratification check:")
for fold in range(N_FOLDS):
    fold_data = [row for row in fold_assay_dist if row['fold'] == fold]
    logging.info(f"Fold {fold} has {len(fold_data)} distinct assay-value combinations")

# =============================================================================
# CREATE CROSS-VALIDATION SPLITS AND TENSOR DATASETS
# =============================================================================

def create_tensors_separated(partition, outdir):
    """Create tensor batches with separated components (selfies, properties, values)."""
    try:
        partition = list(partition)
        if not partition:
            logging.warning("Empty partition encountered")
            return
            
        selfies = torch.stack([torch.LongTensor(r.encoded_selfies) for r in partition])
        
        # Keep properties and values separate
        properties_list = []
        values_list = []
        
        for r in partition:
            props = torch.LongTensor([pair.assay_index for pair in r.assay_val_pairs])
            vals = torch.LongTensor([pair.value for pair in r.assay_val_pairs])
            properties_list.append(props)
            values_list.append(vals)
        
        # Pad sequences
        properties = torch.nn.utils.rnn.pad_sequence(properties_list, batch_first=True, padding_value=-1)
        values = torch.nn.utils.rnn.pad_sequence(values_list, batch_first=True, padding_value=-1)
        
        filename = f"{uuid.uuid4()}.pt"
        filepath = (outdir / filename).as_posix()
        torch.save({
            'selfies': selfies, 
            'properties': properties, 
            'values': values
        }, filepath)
        
        logging.info(f"Saved batch with {len(partition)} examples to {filename}")
        
    except Exception as e:
        logging.error(f"Error processing partition: {str(e)}")
        raise

# Process each fold combination
for test_fold in range(N_FOLDS):
    logging.info(f"\n{'='*60}")
    logging.info(f"Processing fold {test_fold} as test set")
    logging.info(f"{'='*60}")
    
    # Define train folds (all except test fold)
    train_folds = [i for i in range(N_FOLDS) if i != test_fold]
    
    # Separate train and test data
    train_data = data_with_folds.filter(F.col("fold").isin(train_folds))
    test_data = data_with_folds.filter(F.col("fold") == test_fold)
    
    train_count_orig = train_data.count()
    test_count = test_data.count()
    
    logging.info(f"Fold {test_fold} - Original training set size: {train_count_orig}")
    logging.info(f"Fold {test_fold} - Test set size: {test_count}")
    
    # Verify class distribution in test set
    test_class_dist = test_data.groupBy("assay_index", "value").count() \
        .filter(F.col("count") < MIN_EXAMPLES_PER_CLASS // N_FOLDS).collect()
    
    if test_class_dist:
        logging.warning(f"Fold {test_fold} - Warning: {len(test_class_dist)} classes have very few test examples")
        for row in test_class_dist[:5]:  # Show first 5
            logging.warning(f"  Assay {row['assay_index']}, Value {row['value']}: {row['count']} test examples")
    
    # For training data, explode the alternative encoded SELFIES to create augmented samples
    train_augmented = train_data.select(
        F.explode(F.col("alternative_encoded_selfies")).alias("encoded_selfies"),
        "assay_index",
        "value"
    ).withColumn("split", F.lit("train"))
    
    train_count_aug = train_augmented.count()
    logging.info(f"Fold {test_fold} - Augmented training set size: {train_count_aug}")
    
    # For test data, keep original encoded_selfies (no augmentation)
    test_data_clean = test_data.select(
        "encoded_selfies", 
        "assay_index", 
        "value"
    ).withColumn("split", F.lit("test"))
    
    # Calculate augmentation factor
    augmentation_factor = train_count_aug / train_count_orig if train_count_orig > 0 else 1
    logging.info(f"Fold {test_fold} - Training set augmentation factor: {augmentation_factor:.2f}x")
    
    # Group data for tensor creation
    train_grouped = train_augmented.groupby("encoded_selfies") \
        .agg(F.collect_list(F.struct("assay_index", "value")).alias("assay_val_pairs"))
    
    test_grouped = test_data_clean.groupby("encoded_selfies") \
        .agg(F.collect_list(F.struct("assay_index", "value")).alias("assay_val_pairs"))
    
    # Create tensor datasets for this fold
    # Training set
    train_output_dir = cvae.utils.mk_empty_directory(
        f'cache/build_tensordataset/cv_tensors_augmented/fold_{test_fold}/train', 
        overwrite=True
    )
    
    train_unique_count = train_grouped.count()
    logging.info(f'Fold {test_fold} - Training set has {train_unique_count} unique SELFIES molecules')
    train_grouped.foreachPartition(lambda part: create_tensors_separated(part, train_output_dir))
    
    # Test set
    test_output_dir = cvae.utils.mk_empty_directory(
        f'cache/build_tensordataset/cv_tensors_augmented/fold_{test_fold}/test', 
        overwrite=True
    )
    
    test_unique_count = test_grouped.count()
    logging.info(f'Fold {test_fold} - Test set has {test_unique_count} unique SELFIES molecules')
    test_grouped.foreachPartition(lambda part: create_tensors_separated(part, test_output_dir))
    
    logging.info(f'Completed building tensors for fold {test_fold}')
    
    # Log fold summary
    logging.info(f"\nFold {test_fold} Summary:")
    logging.info(f"  Train: {train_count_orig} original -> {train_count_aug} augmented ({train_unique_count} unique)")
    logging.info(f"  Test:  {test_count} examples ({test_unique_count} unique)")

# =============================================================================
# BUILD FINAL TRAINING SET (ALL DATA, NO SPLITS)
# =============================================================================

logging.info('\n' + '='*60)
logging.info('Building final training set with augmentation (no splits)')
logging.info('='*60)

# For final training, use all data with augmentation
final_all_data = data_filtered.select('alternative_encoded_selfies', 'assay_index', 'value')

# Explode pre-encoded alternatives for all data
final_augmented = final_all_data.select(
    F.explode(F.col("alternative_encoded_selfies")).alias("encoded_selfies"),
    "assay_index",
    "value"
)

# Group for tensor creation
final_grouped = final_augmented.groupby('encoded_selfies') \
 .agg(F.collect_list(F.struct('assay_index', 'value')).alias('assay_val_pairs')) \
 .cache()

final_output_dir = cvae.utils.mk_empty_directory(
    'cache/build_tensordataset/final_tensors_augmented', 
    overwrite=True
)

final_count = final_grouped.count()
logging.info(f'Final augmented dataset has {final_count} unique SELFIES molecules')

final_grouped.foreachPartition(lambda part: create_tensors_separated(part, final_output_dir))

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

logging.info("\n" + "="*60)
logging.info("FINAL 5-FOLD CV DATASET SUMMARY")
logging.info("="*60)

# Overall fold statistics
logging.info(f"\nCross-validation configuration:")
logging.info(f"  Number of folds: {N_FOLDS}")
logging.info(f"  Train/test split ratio: {(N_FOLDS-1)}/{N_FOLDS} : 1/{N_FOLDS} (~{100*(N_FOLDS-1)/N_FOLDS:.0f}%/{100/N_FOLDS:.0f}%)")

# Per-fold statistics
logging.info(f"\nPer-fold data distribution:")
for row in fold_counts:
    pct = (row['count'] / sum(fc['count'] for fc in fold_counts)) * 100
    logging.info(f"  Fold {row['fold']}: {row['count']} examples ({pct:.1f}%)")

# Verify minimum examples per class in each fold
logging.info(f"\nVerifying minimum examples per class-fold combination:")
fold_class_min = data_with_folds.groupBy("fold", "assay_index", "value") \
    .count().groupBy("fold").agg(F.min("count").alias("min_count")).orderBy("fold").collect()

for row in fold_class_min:
    expected_min = MIN_EXAMPLES_PER_CLASS // N_FOLDS
    status = "✓" if row['min_count'] >= expected_min else "⚠"
    logging.info(f"  Fold {row['fold']}: minimum {row['min_count']} examples per class {status}")

# Find longest assay-val sequence across all folds
max_seq_query = data_with_folds.groupby("encoded_selfies") \
    .agg(F.collect_list(F.struct("assay_index", "value")).alias("assay_val_pairs")) \
    .withColumn("seq_length", F.size("assay_val_pairs"))

max_assay_val_count = max_seq_query.agg(F.max("seq_length")).collect()[0][0]
avg_assay_val_count = max_seq_query.agg(F.avg("seq_length")).collect()[0][0]

logging.info(f"\nSequence statistics:")
logging.info(f"  Longest assay-value sequence: {max_assay_val_count}")
logging.info(f"  Average assay-values per example: {avg_assay_val_count:.2f}")

# Final summary
total_original = data_filtered.count()
total_augmented_estimate = total_original * augmentation_factor  # Using last fold's factor as estimate

logging.info(f"\n{'='*60}")
logging.info("AUGMENTATION SUMMARY")
logging.info(f"{'='*60}")
logging.info(f"✓ Created {N_FOLDS}-fold cross-validation splits")
logging.info(f"✓ Each fold serves as test set once, with remaining {N_FOLDS-1} folds as training")
logging.info(f"✓ Training sets augmented by ~{augmentation_factor:.2f}x")
logging.info(f"✓ Test sets kept unchanged (no augmentation)")
logging.info(f"✓ Original dataset: {total_original} examples")
logging.info(f"✓ Estimated augmented training size: ~{int(total_augmented_estimate)} examples per fold")
logging.info(f"✓ All tensor datasets created successfully in cache/build_tensordataset/cv_tensors_augmented/")

logging.info("\n5-Fold CV augmented dataset build completed successfully!")
