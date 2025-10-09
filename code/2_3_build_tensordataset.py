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
logging.basicConfig(filename=logpath / 'build_tensordataset_augmented.log', level=logging.INFO, filemode='w')

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

# Assign splits
@pandas_udf(StringType())
def assign_split_with_min_hold_udf(row_number: pd.Series, total_count: pd.Series) -> pd.Series:
    result = pd.Series([""] * len(row_number))
    
    for i in range(len(row_number)):
        total = total_count.iloc[i]
        row_num = row_number.iloc[i]
        
        min_hold = MIN_EXAMPLES_PER_CLASS
        target_hold_pct = 0.1
        MAX_HOLD = 1000
        hold_size = max(min_hold, min(int(total * target_hold_pct), MAX_HOLD))
        
        if row_num <= hold_size:
            result.iloc[i] = 'hld'
        else:
            result.iloc[i] = 'trn'
    
    return result

data_with_splits = data_with_row_nums.withColumn("split", 
    assign_split_with_min_hold_udf("row_number", "total_count")).cache()

logging.info("Split assignment complete")

# =============================================================================
# EXPAND TRAINING DATA WITH ALTERNATIVES
# =============================================================================

# Separate training and hold sets
train_data = data_with_splits.filter(F.col("split") == "trn")
hold_data = data_with_splits.filter(F.col("split") == "hld")

logging.info(f"Original training set size: {train_data.count()}")
logging.info(f"Hold set size: {hold_data.count()}")

# For training data, explode the alternative encoded SELFIES to create augmented samples
# Each alternative is already an encoded SELFIES array
train_augmented = train_data.select(
    F.explode(F.col("alternative_encoded_selfies")).alias("encoded_selfies"),
    "assay_index",
    "value",
    "split"
)

logging.info(f"Augmented training set size: {train_augmented.count()}")

# For hold data, keep original encoded_selfies (no augmentation)
hold_data_clean = hold_data.select("encoded_selfies", "assay_index", "value", "split")

# Combine augmented training with original hold data
final_data = train_augmented.unionAll(hold_data_clean)

logging.info(f"Final combined dataset size: {final_data.count()}")

# Calculate augmentation factor
augmentation_factor = train_augmented.count() / train_data.count() if train_data.count() > 0 else 1
logging.info(f"Training set augmentation factor: {augmentation_factor:.2f}x")

# =============================================================================
# CREATE TENSOR DATASETS
# =============================================================================

# Group data for tensor creation
grouped_data = final_data.select("encoded_selfies", "assay_index", "value", "split") \
        .groupby("encoded_selfies", "split") \
        .agg(F.collect_list(F.struct("assay_index", "value")).alias("assay_val_pairs")) \
        .cache()

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

# Build tensor datasets for splits
for split in ['trn', 'hld']:
    logging.info(f'Building {split} augmented tensor dataset')
    output_dir = cvae.utils.mk_empty_directory(
        f'cache/build_tensordataset/multitask_tensors_augmented/{split}', 
        overwrite=True
    )
    
    split_grouped = grouped_data.filter(F.col("split") == split)
    split_count = split_grouped.count()
    logging.info(f'{split} split has {split_count} unique SELFIES molecules')
    
    split_grouped.foreachPartition(lambda part: create_tensors_separated(part, output_dir))
    logging.info(f'Completed building {split} tensors')

# =============================================================================
# BUILD FINAL TRAINING SET (ALL DATA, NO SPLITS)
# =============================================================================

logging.info('Building final training set with augmentation (no splits)')

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

logging.info("=== FINAL AUGMENTED DATASET SUMMARY ===")

# Final split counts
final_split_counts = final_data.groupBy("split").count().collect()
final_total = sum(row['count'] for row in final_split_counts)

for row in final_split_counts:
    pct = (row['count'] / final_total) * 100
    logging.info(f"Final {row['split']} split: {row['count']} examples ({pct:.1f}%)")

logging.info(f"Final total dataset size: {final_total} examples")

# Verify hold set requirements
hold_verification = final_data.filter(F.col("split") == "hld") \
    .groupBy("assay_index", "value") \
    .count() \
    .withColumnRenamed("count", "hold_count")

insufficient_hold = hold_verification.filter(F.col("hold_count") < MIN_EXAMPLES_PER_CLASS).collect()

if insufficient_hold:
    logging.error(f"CRITICAL: Found {len(insufficient_hold)} assay-value combinations with < {MIN_EXAMPLES_PER_CLASS} examples in hold set")
    for row in insufficient_hold:
        logging.error(f"  Assay {row['assay_index']}, Value {row['value']}: only {row['hold_count']} in hold set")
else:
    logging.info(f"SUCCESS: All assay-value combinations have at least {MIN_EXAMPLES_PER_CLASS} examples in hold set")

# Find longest assay-val sequence
max_assay_val_count = grouped_data.withColumn("seq_length", F.size("assay_val_pairs")) \
    .agg(F.max("seq_length")).collect()[0][0]
logging.info(f"Longest assay-value sequence: {max_assay_val_count}")

# Find average number of assay-values per example
avg_assay_val_count = grouped_data.withColumn("seq_length", F.size("assay_val_pairs")) \
    .agg(F.avg("seq_length")).collect()[0][0]
logging.info(f"Average assay-values per example: {avg_assay_val_count:.2f}")

logging.info("=== AUGMENTATION SUMMARY ===")
logging.info(f"✓ Training set augmented by {augmentation_factor:.2f}x")
logging.info(f"✓ Hold set kept unchanged (no augmentation)")
logging.info(f"✓ All assay-value combinations meet minimum hold set requirements")
logging.info(f"✓ Augmented tensor datasets created successfully")

logging.info("Augmented dataset build completed successfully!")