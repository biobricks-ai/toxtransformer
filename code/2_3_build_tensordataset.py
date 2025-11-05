# cmd:
"""
PYTHONPATH=./ spark-submit --master local[240] --driver-memory 512g \
--conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///data/tmp/spark-events \
--conf spark.local.dir=/data/tmp/spark-local code/2_3_build_tensordataset.py; \
./slackmsg 'Build tensordataset finished'
"""

import uuid, torch, torch.nn.utils.rnn
import pyspark.sql.functions as F
import cvae.utils
import logging
import pathlib
from pyspark.sql import Window

# =============================================================================
# SETUP
# =============================================================================

logpath = pathlib.Path('cache/build_tensordataset/log')
logpath.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    filename=logpath / 'build_tensordataset_bootstrap.log', 
    level=logging.INFO, 
    filemode='w'
)

spark = cvae.utils.get_spark_session()

# Configuration
N_BOOTSTRAP = 5
TARGET_TEST_PER_CLASS = 1000     # Aim for 1000 of each
MIN_TEST_PER_CLASS = 20          # Hard requirement: at least 20 of each
TEST_PERCENT = 0.05              # 5% of available examples
MIN_TOTAL_PER_CLASS = MIN_TEST_PER_CLASS + 20  # Need enough for train too
RANDOM_SEED = 42

# =============================================================================
# LOAD AND FILTER DATA
# =============================================================================

data = (spark.read.parquet("cache/preprocess_activities/activities_augmented.parquet")
        .filter(F.col('source') != 'ctdbase')
        .select('smiles', 'encoded_selfies', 'assay_index', 'value')
        .groupby('smiles', 'encoded_selfies', 'assay_index')
        .agg(F.collect_set('value').alias('values'))
        .filter(F.size('values') == 1)
        .select('smiles', 'encoded_selfies', 'assay_index', 
                F.element_at('values', 1).alias('value')))

logging.info("Initial data loaded")

# Filter to property-value pairs with sufficient examples
assay_value_counts = data.groupBy("assay_index", "value").count()
valid_pairs = assay_value_counts.filter(F.col("count") >= MIN_TOTAL_PER_CLASS)

data_filtered = data.join(valid_pairs, on=["assay_index", "value"]).cache()

total_count = data_filtered.count()
logging.info(f"Filtered to {total_count} examples meeting minimum class size")

# Determine which properties have both positive and negative classes
properties_with_both = (valid_pairs
                       .groupBy("assay_index")
                       .agg(F.collect_set("value").alias("values"))
                       .filter(F.size("values") == 2)
                       .select("assay_index"))

data_balanced = data_filtered.join(properties_with_both, on="assay_index").cache()
balanced_count = data_balanced.count()

logging.info(f"Properties with both classes: {properties_with_both.count()}")
logging.info(f"Examples from balanced properties: {balanced_count}")

# Log class distribution
class_dist = (data_balanced
              .groupBy("assay_index", "value")
              .count()
              .orderBy("assay_index", "value")
              .collect())

logging.info("\nClass distribution (first 20):")
for row in class_dist[:20]:
    logging.info(f"  Property {row['assay_index']}, Value {row['value']}: {row['count']} examples")

# =============================================================================
# TENSOR CREATION HELPER
# =============================================================================

def create_tensors_separated(partition, outdir):
    """Create tensor batches with separated components."""
    partition = list(partition)
    if not partition:
        return
    
    selfies_list = [torch.LongTensor(r.encoded_selfies) for r in partition]
    selfies = torch.nn.utils.rnn.pad_sequence(selfies_list, batch_first=True, padding_value=0)
    
    properties_list = [torch.LongTensor([p.assay_index for p in r.assay_val_pairs]) for r in partition]
    values_list = [torch.LongTensor([p.value for p in r.assay_val_pairs]) for r in partition]
    
    properties = torch.nn.utils.rnn.pad_sequence(properties_list, batch_first=True, padding_value=-1)
    values = torch.nn.utils.rnn.pad_sequence(values_list, batch_first=True, padding_value=-1)
    
    filepath = (outdir / f"{uuid.uuid4()}.pt").as_posix()
    torch.save({'selfies': selfies, 'properties': properties, 'values': values}, filepath)

# =============================================================================
# CREATE BOOTSTRAP LEAVE-N-OUT SPLITS
# =============================================================================

for bootstrap_idx in range(N_BOOTSTRAP):
    logging.info(f"\n{'='*60}")
    logging.info(f"Creating bootstrap split {bootstrap_idx}")
    logging.info(f"{'='*60}")
    
    # Calculate test size: min(1000, 5% of available, but at least 20)
    seed = RANDOM_SEED + bootstrap_idx
    window = Window.partitionBy('assay_index', 'value').orderBy(F.rand(seed=seed))
    
    split_data = (data_balanced
                  .withColumn('row_num', F.row_number().over(window))
                  .withColumn('total_in_class', F.count('*').over(Window.partitionBy('assay_index', 'value')))
                  # Calculate test size: min(TARGET, 5% of total, but at least MIN)
                  .withColumn('percent_test', (F.col('total_in_class') * TEST_PERCENT).cast('int'))
                  .withColumn('n_test', 
                              F.greatest(
                                  F.lit(MIN_TEST_PER_CLASS),
                                  F.least(
                                      F.lit(TARGET_TEST_PER_CLASS),
                                      F.col('percent_test')
                                  )
                              ))
                  .withColumn('is_test', F.col('row_num') <= F.col('n_test'))
                  .cache())
    
    train_data = split_data.filter(~F.col('is_test'))
    test_data = split_data.filter(F.col('is_test'))
    
    train_count = train_data.count()
    test_count = test_data.count()
    
    logging.info(f"Split {bootstrap_idx}: Train={train_count}, Test={test_count}")
    
    # Check test set statistics per class
    test_class_counts = (test_data
                        .groupBy("assay_index", "value")
                        .count()
                        .orderBy("assay_index", "value"))
    
    test_stats = test_class_counts.agg(
        F.min("count").alias("min"),
        F.max("count").alias("max"),
        F.avg("count").alias("avg")
    ).collect()[0]
    
    logging.info(f"  Test set per class - Min: {test_stats['min']}, Max: {test_stats['max']}, Avg: {test_stats['avg']:.1f}")
    
    # Check balance per property (positives vs negatives)
    test_balance = (test_data
                   .groupBy("assay_index")
                   .agg(F.sum(F.when(F.col("value") == 1, 1).otherwise(0)).alias("positives"),
                        F.sum(F.when(F.col("value") == 0, 1).otherwise(0)).alias("negatives")))
    
    balance_stats = test_balance.agg(
        F.min("positives").alias("min_pos"),
        F.max("positives").alias("max_pos"),
        F.min("negatives").alias("min_neg"),
        F.max("negatives").alias("max_neg")
    ).collect()[0]
    
    logging.info(f"  Test balance - Positives: [{balance_stats['min_pos']}, {balance_stats['max_pos']}], "
                f"Negatives: [{balance_stats['min_neg']}, {balance_stats['max_neg']}]")
    
    # Show first 10 properties with their test allocation details
    test_allocation = (split_data
                      .select('assay_index', 'value', 'total_in_class', 'n_test')
                      .distinct()
                      .orderBy('assay_index', 'value')
                      .collect())
    
    logging.info(f"  Test set allocation (first 10 classes):")
    for row in test_allocation[:10]:
        percent = (row['n_test'] / row['total_in_class']) * 100
        logging.info(f"    Property {row['assay_index']}, Value {row['value']}: "
                    f"{row['n_test']}/{row['total_in_class']} ({percent:.1f}%)")
    
    # Show first 10 properties balance
    balance_check = test_balance.collect()
    logging.info(f"  Test set balance (first 10 properties):")
    for row in balance_check[:10]:
        logging.info(f"    Property {row['assay_index']}: {row['positives']} pos, {row['negatives']} neg")
    
    # Group by molecule for tensor creation
    train_grouped = (train_data
                     .select('encoded_selfies', 'assay_index', 'value')
                     .groupby('encoded_selfies')
                     .agg(F.collect_list(F.struct('assay_index', 'value')).alias('assay_val_pairs')))
    
    test_grouped = (test_data
                    .select('encoded_selfies', 'assay_index', 'value')
                    .groupby('encoded_selfies')
                    .agg(F.collect_list(F.struct('assay_index', 'value')).alias('assay_val_pairs')))
    
    # Save tensors
    train_dir = cvae.utils.mk_empty_directory(
        f'cache/build_tensordataset/bootstrap/split_{bootstrap_idx}/train', 
        overwrite=True
    )
    test_dir = cvae.utils.mk_empty_directory(
        f'cache/build_tensordataset/bootstrap/split_{bootstrap_idx}/test', 
        overwrite=True
    )
    
    train_grouped.foreachPartition(lambda p: create_tensors_separated(p, train_dir))
    test_grouped.foreachPartition(lambda p: create_tensors_separated(p, test_dir))
    
    logging.info(f"Completed bootstrap split {bootstrap_idx}")
    
    split_data.unpersist()

# =============================================================================
# BUILD FINAL TRAINING SET
# =============================================================================

logging.info('\n' + '='*60)
logging.info('Building final training set (all data)')
logging.info('='*60)

final_grouped = (data_balanced
                 .select('encoded_selfies', 'assay_index', 'value')
                 .groupby('encoded_selfies')
                 .agg(F.collect_list(F.struct('assay_index', 'value')).alias('assay_val_pairs')))

final_dir = cvae.utils.mk_empty_directory('cache/build_tensordataset/final_tensors', overwrite=True)
final_count = final_grouped.count()

logging.info(f'Final dataset: {final_count} unique molecules')
final_grouped.foreachPartition(lambda p: create_tensors_separated(p, final_dir))

# =============================================================================
# SUMMARY
# =============================================================================

logging.info("\n" + "="*60)
logging.info("BOOTSTRAP LEAVE-N-OUT DATASET SUMMARY")
logging.info("="*60)
logging.info(f"✓ Created {N_BOOTSTRAP} bootstrap splits")
logging.info(f"✓ Test set target: min({TARGET_TEST_PER_CLASS}, {TEST_PERCENT*100}% of class, but ≥{MIN_TEST_PER_CLASS})")
logging.info(f"✓ Only includes properties with both positive and negative examples")
logging.info(f"✓ Total examples: {balanced_count}")
logging.info(f"✓ Unique molecules: {final_count}")
logging.info("\nBootstrap leave-N-out dataset build completed successfully!")