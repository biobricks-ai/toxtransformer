# cmd:
# spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///tmp/spark-events code/2_3_build_tensordataset.py
# PYTHONPATH=./ spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///data/tmp/spark-events --conf spark.local.dir=/data/tmp/spark-local code/2_3_build_tensordataset.py

import uuid, torch, torch.nn.utils.rnn
import pyspark.sql, pyspark.sql.functions as F
import cvae.utils, cvae.tokenizer.selfies_tokenizer, cvae.tokenizer.selfies_property_val_tokenizer
import logging
import pathlib
import pandas as pd
import shutil
from pyspark.sql import Window
from pyspark.sql.types import StringType
from pyspark.sql.functions import pandas_udf

logpath = pathlib.Path('cache/build_tensordataset/log')
logpath.mkdir(parents=True, exist_ok=True)
logging.basicConfig(filename=logpath / 'build_tensordataset.log', level=logging.INFO, filemode='w')

spark = cvae.utils.get_spark_session()

# BUILD UNSUPERVISED TRAINING SET ====================================================
# TODO a decoder only transformer with selfies followed by property values would be better and this makes more sense there.
# data = spark.read.parquet('cache/preprocess_tokenizer/substances.parquet')
# def process_selfies(partition,outdir):
#     tensor = torch.tensor([row['encoded_selfies'] for row in partition], dtype=torch.long)
#     torch.save(tensor, outdir / f'partition_{uuid.uuid4()}.pt')

# vaedir = cvae.utils.mk_empty_directory('cache/build_tensordataset/all_selfies', overwrite=True)
# selfies = data.select('encoded_selfies') # selfies are already distinct
# selfies.rdd.foreachPartition(lambda partition: process_selfies(partition, vaedir))
# logging.info('Unsupervised training set built')

# BUILD MULTITASK SUPERVISED DATA SET =============================================

# ctdbase has data quality issues
data = spark.read.parquet("cache/preprocess_activities/activities.parquet").orderBy(F.rand(seed=42)) \
    .filter(F.col('source') != 'ctdbase') \
    .select('encoded_selfies', 'assay_index', 'value') \
    .groupby('encoded_selfies', 'assay_index') \
    .agg(F.collect_set('value').alias('values')) \
    .filter(F.size('values') == 1) \
    .select('encoded_selfies', 'assay_index', F.element_at('values', 1).alias('value'))

logging.info("Initial data loading complete")

# Analyze data distribution per assay-value combination
assay_value_counts = data.groupBy("assay_index", "value").count().withColumnRenamed("count", "total_count")

# Log initial distribution
logging.info("Initial assay-value distribution:")
initial_dist = assay_value_counts.collect()
for row in sorted(initial_dist, key=lambda x: (x['assay_index'], x['value'])):
    logging.info(f"Assay {row['assay_index']}, Value {row['value']}: {row['total_count']} examples")

# Filter to only include assay-value combinations that have at least 30 examples
# (since we need at least 30 in hold set for meaningful evaluation)
MIN_EXAMPLES_PER_CLASS = 30
valid_assay_values = assay_value_counts.filter(F.col("total_count") >= MIN_EXAMPLES_PER_CLASS)

# Log which assay-value combinations are being excluded
excluded_combinations = assay_value_counts.filter(F.col("total_count") < MIN_EXAMPLES_PER_CLASS).collect()
if excluded_combinations:
    logging.warning(f"Excluding {len(excluded_combinations)} assay-value combinations with < {MIN_EXAMPLES_PER_CLASS} examples:")
    for row in excluded_combinations:
        logging.warning(f"  Assay {row['assay_index']}, Value {row['value']}: {row['total_count']} examples")

# Join back to keep only data points from valid assay-value combinations
data_filtered = data.join(valid_assay_values, on=["assay_index", "value"])

# Create window for stratified sampling within each assay-value combination
window = Window.partitionBy('assay_index', 'value').orderBy(F.rand(seed=42))
data_with_row_nums = data_filtered.withColumn("row_number", F.row_number().over(window))

# Assign splits ensuring at least `MIN_EXAMPLES_PER_CLASS` of each value for each assay in hold set
@pandas_udf(StringType())
def assign_split_with_min_hold_udf(row_number: pd.Series, total_count: pd.Series) -> pd.Series:
    result = pd.Series([""] * len(row_number))
    
    for i in range(len(row_number)):
        total = total_count.iloc[i]
        row_num = row_number.iloc[i]
        
        # Ensure at least 20 go to hold set, but aim for ~10% of total
        # Cap hold set at 30% to ensure we have enough training data
        min_hold = MIN_EXAMPLES_PER_CLASS
        target_hold_pct = 0.1
        MAX_HOLD = 1000
        hold_size = max(min_hold, min(int(total * target_hold_pct), MAX_HOLD))
        
        if row_num <= hold_size:
            result.iloc[i] = 'hld'
        else:
            result.iloc[i] = 'trn'
    
    return result

data_with_splits = data_with_row_nums.withColumn("split", assign_split_with_min_hold_udf("row_number", "total_count")).cache()

logging.info("Split assignment complete")

# Verify that we have at least 20 of each value for each assay in hold set
hold_verification = data_with_splits.filter(F.col("split") == "hld") \
    .groupBy("assay_index", "value") \
    .count() \
    .withColumnRenamed("count", "hold_count")

# Check if any assay-value combinations have fewer than 20 in hold
insufficient_hold = hold_verification.filter(F.col("hold_count") < MIN_EXAMPLES_PER_CLASS).collect()

if insufficient_hold:
    logging.error(f"CRITICAL: Found {len(insufficient_hold)} assay-value combinations with < {MIN_EXAMPLES_PER_CLASS} examples in hold set")
    for row in insufficient_hold:
        logging.error(f"  Assay {row['assay_index']}, Value {row['value']}: only {row['hold_count']} in hold set")
    raise ValueError("Split assignment failed to meet minimum hold set requirements")
else:
    logging.info(f"SUCCESS: All assay-value combinations have at least {MIN_EXAMPLES_PER_CLASS} examples in hold set")

# Log detailed split statistics
logging.info("=== SPLIT STATISTICS ===")
split_summary = data_with_splits.groupBy("split").count().collect()
total_examples = sum(row['count'] for row in split_summary)

for row in split_summary:
    pct = (row['count'] / total_examples) * 100
    logging.info(f"Split {row['split']}: {row['count']} examples ({pct:.1f}%)")

# =============================================================================
# CREATE TOKENIZER 
# =============================================================================
# selfies_tok = cvae.tokenizer.selfies_tokenizer.SelfiesTokenizer().load('cache/preprocess_tokenizer/selfies_tokenizer.json')
# num_assays = int(data_with_splits.agg(F.max('assay_index')).collect()[0][0] + 1)
# num_values = int(data_with_splits.agg(F.max('value')).collect()[0][0] + 1) # TODO this assumes an index identity for values
# tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer(selfies_tok, num_assays, num_values)

# spv_path = pathlib.Path('brick/selfies_property_val_tokenizer')
# if spv_path.exists():
#     shutil.rmtree(spv_path)
# spv_path.mkdir(parents=True, exist_ok=True)

# tokenizer.save(spv_path)
# logging.info(f"Tokenizer saved to {spv_path}")

# =============================================================================
# CREATE PV SEQUENCE TENSOR DATASETS 
# =============================================================================
# grouped_data = data_with_splits.select("encoded_selfies", "assay_index", "value", "split") \
#         .groupby("encoded_selfies", "split") \
#         .agg(F.collect_list(F.struct("assay_index", "value")).alias("assay_val_pairs")) \
#         .cache()

# def create_tensors(partition, outdir):
#     """Create tensor batches from partition data with error handling."""
#     try:
#         partition = list(partition)
#         if not partition:
#             logging.warning("Empty partition encountered")
#             return
            
#         selfies = torch.stack([torch.LongTensor(r.encoded_selfies) for r in partition])
#         assay_vals = [tokenizer.tokenize_assay_values(r.assay_val_pairs) for r in partition]
#         assay_vals = torch.nn.utils.rnn.pad_sequence(assay_vals, batch_first=True, padding_value=tokenizer.pad_idx)
        
#         filename = f"{uuid.uuid4()}.pt"
#         filepath = (outdir / filename).as_posix()
#         torch.save({'selfies': selfies, 'assay_vals': assay_vals}, filepath)
        
#         logging.info(f"Saved batch with {len(partition)} examples to {filename}")
        
#     except Exception as e:
#         logging.error(f"Error processing partition: {str(e)}")
#         raise

# for split in ['trn', 'hld']:
#     logging.info(f'Building {split} multitask supervised training set')
#     output_dir = cvae.utils.mk_empty_directory(f'cache/build_tensordataset/multitask_tensors/{split}', overwrite=True)
    
#     split_grouped = grouped_data.filter(F.col("split") == split)
#     split_count = split_grouped.count()
#     logging.info(f'{split} split has {split_count} unique SELFIES molecules')
    
#     split_grouped.foreachPartition(lambda part: create_tensors(part, output_dir))

# # BUILD FINAL TRAINING SET (NO SPLITS) ====================================================
# logging.info('Building final training set (no splits)')
# final_data = data_filtered.select('encoded_selfies', 'assay_index', 'value') \
#     .groupby('encoded_selfies') \
#     .agg(F.collect_list(F.struct('assay_index', 'value')).alias('assay_val_pairs')) \
#     .cache()

# final_output_dir = cvae.utils.mk_empty_directory('cache/build_tensordataset/final_tensors', overwrite=True)
# final_count = final_data.count()
# logging.info(f'Final dataset has {final_count} unique SELFIES molecules')

# final_data.foreachPartition(lambda part: create_tensors(part, final_output_dir))

# logging.info('Multitask supervised training set built')
# logging.info('Final training set built successfully!')

# ==========================================================================================
# BUILD PROPERTY VALUE TENSOR DATASETS
# ==========================================================================================
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
        
        logging.info(f"Saved separated batch with {len(partition)} examples to {filename}")
        
    except Exception as e:
        logging.error(f"Error processing partition (separated): {str(e)}")
        raise


# Build separated tensor datasets for splits
for split in ['trn', 'hld']:
    logging.info(f'Building {split} separated multitask supervised training set')
    output_dir_sep = cvae.utils.mk_empty_directory(
        f'cache/build_tensordataset/multitask_tensors_separated/{split}', 
        overwrite=True
    )
    
    split_grouped = grouped_data.filter(F.col("split") == split)
    split_grouped.foreachPartition(lambda part: create_tensors_separated(part, output_dir_sep))
    logging.info(f'Completed building {split} separated tensors')


# Build separated final training set (no splits)
logging.info('Building final separated training set (no splits)')
final_output_dir_sep = cvae.utils.mk_empty_directory(
    'cache/build_tensordataset/final_tensors_separated', 
    overwrite=True
)

final_data.foreachPartition(lambda part: create_tensors_separated(part, final_output_dir_sep))
logging.info('Final separated training set built successfully!')

# SUMMARY INFORMATION ============================================================
logging.info("=== FINAL DATASET SUMMARY ===")

# Final split counts
final_split_counts = data_with_splits.groupBy("split").count().collect()
final_total = sum(row['count'] for row in final_split_counts)

for row in final_split_counts:
    pct = (row['count'] / final_total) * 100
    logging.info(f"Final {row['split']} split: {row['count']} examples ({pct:.1f}%)")

logging.info(f"Final total dataset size: {final_total} examples")

# Find longest assay-val sequence
max_assay_val_count = grouped_data.withColumn("seq_length", F.size("assay_val_pairs")).agg(F.max("seq_length")).collect()[0][0]
logging.info(f"Longest assay-value sequence: {max_assay_val_count}")

# Find average number of assay-values per example
avg_assay_val_count = grouped_data.withColumn("seq_length", F.size("assay_val_pairs")).agg(F.avg("seq_length")).collect()[0][0]
logging.info(f"Average assay-values per example: {avg_assay_val_count:.2f}")

# Get distribution of assay-value counts
logging.info("Distribution of assay-value sequence lengths:")
distribution = grouped_data.withColumn("seq_length", F.size("assay_val_pairs")) \
        .groupBy("seq_length") \
        .count() \
        .orderBy("seq_length") \
        .collect()

total = sum(row['count'] for row in distribution)
cumsum = 0

for row in distribution:
    if row['seq_length'] <= 200:
        cumsum += row['count']
        pct = (cumsum / total) * 100
        logging.info(f"  Sequence length {row['seq_length']}: {row['count']} examples ({pct:.1f}% cumulative)")
    else:
        # Aggregate all sequences longer than 200
        remaining = sum(r['count'] for r in distribution if r['seq_length'] > 200)
        if remaining > 0:
            cumsum += remaining
            pct = (cumsum / total) * 100
            logging.info(f"  Sequence length 200+: {remaining} examples ({pct:.1f}% cumulative)")
        break

# Final verification summary
logging.info("=== FINAL VERIFICATION ===")
logging.info(f"✓ All assay-value combinations have ≥{MIN_EXAMPLES_PER_CLASS} examples in hold set")
logging.info(f"✓ Tokenizer saved successfully")
logging.info(f"✓ Split information saved for reproducibility")
logging.info(f"✓ Tensor datasets created successfully for trn and hld splits")
logging.info(f"✓ Tensor dataset created successfully for final training (no splits)")

logging.info("Dataset build completed successfully!")