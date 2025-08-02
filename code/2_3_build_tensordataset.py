# cmd:
# spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///tmp/spark-events code/2_3_build_tensordataset.py
# spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///data/tmp/spark-events --conf spark.local.dir=/data/tmp/spark-local code/2_3_build_tensordataset.py

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

data = spark.read.parquet("cache/preprocess_activities/activities.parquet").orderBy(F.rand(seed=42)) \
    .select('encoded_selfies', 'assay_index', 'value') \
    .groupby('encoded_selfies', 'assay_index') \
    .agg(F.collect_set('value').alias('values')) \
    .filter(F.size('values') == 1) \
    .select('encoded_selfies', 'assay_index', F.element_at('values', 1).alias('value')) \
    .cache()

# Create train/test/hold splits (89/1/10) stratified by assay
window = Window.partitionBy('assay_index', 'value').orderBy(F.rand(seed=42))
avcounts = data.groupBy("assay_index", "value").count().withColumnRenamed("count", "n_total")
data = data.withColumn("row_number", F.row_number().over(window)).join(avcounts, on=["assay_index", "value"])

# Use pandas_udf for split assignment
@pandas_udf(StringType())
def assign_split_udf(row_number: pd.Series, n_total: pd.Series) -> pd.Series:
    result = pd.Series([""] * len(row_number))
    ratio = row_number / n_total
    result[ratio <= 0.89] = 'trn'
    result[(ratio > 0.94) & (ratio <= 0.95)] = 'tst'
    result[ratio > 0.95] = 'hld'
    return result

data = data.withColumn("split", assign_split_udf("row_number", "n_total"))

# Set up tokenizer
selfies_tok = cvae.tokenizer.selfies_tokenizer.SelfiesTokenizer().load('cache/preprocess_tokenizer/selfies_tokenizer.json')
num_assays = int(data.agg(F.max('assay_index')).collect()[0][0] + 1)
num_values = int(data.agg(F.max('value')).collect()[0][0] + 1) # TODO this assumes an index identity for values
tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer(selfies_tok, num_assays, num_values)
spv_path = pathlib.Path('brick/selfies_property_val_tokenizer')
if spv_path.exists():
    shutil.rmtree(spv_path)
spv_path.mkdir(parents=True, exist_ok=True)

tokenizer.save(spv_path)

# Group data by SELFIES and split for batch processing
grouped_data = data.select("encoded_selfies", "assay_index", "value", "split") \
        .groupby("encoded_selfies", "split") \
        .agg(F.collect_list(F.struct("assay_index", "value")).alias("assay_val_pairs")) \
        .cache()

def create_tensors(partition, outdir):
    partition = list(partition)
    selfies = torch.stack([torch.LongTensor(r.encoded_selfies) for r in partition])
    assay_vals = [tokenizer.tokenize_assay_values(r.assay_val_pairs) for r in partition]
    assay_vals = torch.nn.utils.rnn.pad_sequence(assay_vals, batch_first=True, padding_value=tokenizer.pad_idx)
    torch.save({'selfies': selfies, 'assay_vals': assay_vals}, (outdir / f"{uuid.uuid4()}.pt").as_posix())

# TODO we need to record which inchi + assay_indexes are in each split
for split in ['trn', 'tst', 'hld']:
    logging.info(f'Building {split} multitask supervised training set')
    output_dir = cvae.utils.mk_empty_directory(f'cache/build_tensordataset/multitask_tensors/{split}', overwrite=True)
    grouped_data.filter(F.col("split") == split).foreachPartition(lambda part: create_tensors(part, output_dir))
        
logging.info('Multitask supervised training set built')

# SUMMARY INFORMATION ============================================================
logging.info("Generating dataset summary statistics...")

# Count examples per split
split_counts = data.groupBy("split").count().collect()
for row in split_counts:
    logging.info(f"Split {row['split']}: {row['count']} examples")

# Calculate total dataset size
total_count = sum(row['count'] for row in split_counts)
logging.info(f"Total dataset size: {total_count} examples")

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
        cumsum += remaining
        pct = (cumsum / total) * 100
        logging.info(f"  Sequence length 200+: {remaining} examples ({pct:.1f}% cumulative)")
        break