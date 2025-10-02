# cmd: 
# spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///tmp/spark-events code/1_3_preprocess_activities.py
# spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///data/tmp/spark-events --conf spark.local.dir=/data/tmp/spark-local code/1_3_preprocess_activities.py

import biobricks
import cvae
import cvae.utils
import logging
import pandas as pd
import pathlib
import pyspark.ml.feature
from pyspark.sql import functions as F
from pyspark.sql.functions import col, from_json, first
from pyspark.sql.types import StructType, StructField, StringType, IntegerType
from pyspark.sql.window import Window

# set up logging
logdir = pathlib.Path('cache/preprocess_activities/log')
logdir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=logdir / 'preprocess_activities.log', filemode='w')

outdir = pathlib.Path('cache/preprocess_activities')
outdir.mkdir(parents=True, exist_ok=True)

# set up spark session
spark = cvae.utils.get_spark_session()

# GET CHEMARHMONY ACTIVITIES ===========================================================================
logging.info("Loading ChemHarmony activities data...")
chemharmony = biobricks.assets('chemharmony')
activities = spark.read.parquet(chemharmony.activities_parquet).select(['source','smiles','pid','sid','binary_value'])
substances = spark.read.parquet('cache/preprocess_tokenizer/substances.parquet')
data = activities.join(substances.select('sid','selfies','encoded_selfies'), 'sid', 'inner')
data = data.withColumnRenamed('pid', 'assay').withColumnRenamed('binary_value', 'value')
data = data.orderBy(F.rand(52)) # Randomly shuffle data with seed 52
data.cache()

initial_count = data.count()

source_counts = data.groupBy('source').count().orderBy('count', ascending=False).toPandas()
logging.info(str(source_counts))
assert min(source_counts['count']) >= 1000

## FILTER TO USEFUL OR CLASSIFIABLE PROPERTIES ============================================================
logging.info("Filtering assays based on number of unique selfies")

assay_counts = data.groupBy('assay').agg(F.countDistinct('selfies').alias('unique_selfies_count'))
assay_counts = assay_counts.filter(F.col('unique_selfies_count') >= 100)
data = data.join(assay_counts, 'assay', 'inner')

logging.info(f"Data size after filtering: {data.count():,} rows ({(data.count() / initial_count * 100):.1f}% of initial)")

# Filter to assays with at least 50 positives and 50 negatives
logging.info("Filtering assays for balanced positive/negative samples")
assay_balance = data.groupBy('assay').agg(
    F.sum(F.when(F.col('value') == 1, 1).otherwise(0)).alias('positive_count'),
    F.sum(F.when(F.col('value') == 0, 1).otherwise(0)).alias('negative_count')
)

# Filter assays with at least 50 positives AND 50 negatives
balanced_assays = assay_balance.filter((F.col('positive_count') >= 50) & (F.col('negative_count') >= 50))
data = data.join(balanced_assays.select('assay'), 'assay', 'inner')

# Check for conflicts (same SELFIES-assay pair with different values)
conflicts = data.groupBy('selfies', 'assay').agg(
    F.countDistinct('value').alias('distinct_values'),
    F.count('*').alias('duplicate_count')
).filter(F.col('distinct_values') > 1)

conflicting_pairs = conflicts.select('selfies', 'assay')
data = data.join(conflicting_pairs, on=['selfies', 'assay'], how='left_anti')

## Map assay UUIDs to integers
logging.info("Converting assay IDs to indices...")
indexer = pyspark.ml.feature.StringIndexer(inputCol="assay", outputCol="assay_index")
data = indexer.fit(data).transform(data)

## write out the processed data, delete activities.parquet if it exists
logging.info("Writing processed data to parquet...")
data.write.parquet((outdir / 'activities.parquet').as_posix(), mode='overwrite')
logging.info(f"wrote {outdir / 'activities.parquet'}")

## TEST count activities ===========================================================
data = spark.read.parquet((outdir / 'activities.parquet').as_posix())
assert data.count() > 10e6 # should have more than 10m activities

source_counts_2 = data.groupBy('source').count().orderBy('count', ascending=False).toPandas()

# Calculate percentage drop for each source
source_counts_2['initial_count'] = source_counts_2['source'].map(source_counts.set_index('source')['count'])
source_counts_2['pct_drop'] = ((source_counts_2['initial_count'] - source_counts_2['count']) / source_counts_2['initial_count'] * 100)
source_counts_2['pct_remaining'] = 100 - source_counts_2['pct_drop']
logging.info("Source counts after filtering:")
logging.info(f"\n{source_counts_2.to_string(index=False, float_format=lambda x: '{:.1f}'.format(x))}")
assert min(source_counts_2['count']) >= 1000