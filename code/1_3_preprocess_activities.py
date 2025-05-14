# cmd: 
# spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///tmp/spark-events code/1_3_preprocess_activities.py

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

initial_count = data.count()

source_counts = data.groupBy('source').count().orderBy('count', ascending=False).toPandas()
logging.info(str(source_counts))
assert min(source_counts['count']) >= 1000

## SPECIAL CTDBASE PROCESSING =============================================================================
# 1. remove ctdbase from data
# 2. extract the GeneSymbol and InteractionActions from the data json field
# 3. set the first assay with each gene_symbol + interaction_action as the assay property
# 4. replace the assay of each row with the assay property


# Define schema for JSON in 'data'
json_schema = StructType([
    StructField("GeneForms", StringType(), True),
    StructField("GeneSymbol", StringType(), True),
    StructField("InteractionActions", StringType(), True),
    StructField("Organism", StringType(), True),
    StructField("OrganismID", IntegerType(), True),
])

# Load properties
properties = spark.read.parquet(chemharmony.properties_parquet).withColumnRenamed('pid', 'assay')

# Extract and normalize ctdbase data
ctdbase = data.filter(col('source') == 'ctdbase') \
              .join(properties, ['assay','source'], 'inner') \
              .withColumn('data_parsed', from_json(col('data'), json_schema)) \
              .withColumn('interaction_action', col('data_parsed.InteractionActions')) \
              .withColumn('gene_symbol', col('data_parsed.GeneSymbol')) \
              .cache()  # Cache for reuse

# Set the first assay per gene_symbol + interaction_action
w = Window.partitionBy('gene_symbol', 'interaction_action').orderBy('assay')
ctdbase = ctdbase.withColumn('representative_assay', F.first('assay').over(w)) \
                 .drop('assay') \
                 .withColumnRenamed('representative_assay', 'assay')

# Combine back into full data
ctdbase = ctdbase.select(data.columns)
data = data.filter(col('source') != 'ctdbase').union(ctdbase)

## FILTER TO USEFUL OR CLASSIFIABLE PROPERTIES ============================================================
logging.info("Filtering assays based on number of unique selfies")

assay_counts = data.groupBy('assay').agg(F.countDistinct('selfies').alias('unique_selfies_count'))
assay_counts = assay_counts.filter(F.col('unique_selfies_count') >= 100)
data = data.join(assay_counts, 'assay', 'inner')

logging.info(f"Data size after filtering: {data.count():,} rows ({(data.count() / initial_count * 100):.1f}% of initial)")

## Map assay UUIDs to integers
logging.info("Converting assay IDs to indices...")
indexer = pyspark.ml.feature.StringIndexer(inputCol="assay", outputCol="assay_index")
data = indexer.fit(data).transform(data)

## write out the processed data, delete activities.parquet if it exists
logging.info("Writing processed data to parquet...")
data.write.parquet((outdir / 'activities.parquet').as_posix(), mode='overwrite')
logging.info(f"wrote {outdir / 'activities.parquet'}")

## TEST count activities ===========================================================
data = spark.read.parquet((outdir / 'activities.parquet').as_posix()).cache()
assert data.count() > 10e6 # should have more than 10m activities

source_counts_2 = data.groupBy('source').count().orderBy('count', ascending=False).toPandas()
# Calculate percentage drop for each source
source_counts_2['initial_count'] = source_counts_2['source'].map(source_counts.set_index('source')['count'])
source_counts_2['pct_drop'] = ((source_counts_2['initial_count'] - source_counts_2['count']) / source_counts_2['initial_count'] * 100)
source_counts_2['pct_remaining'] = 100 - source_counts_2['pct_drop']
logging.info("Source counts after filtering:")
logging.info(f"\n{source_counts_2.to_string(index=False, float_format=lambda x: '{:.1f}'.format(x))}")
assert min(source_counts_2['count']) >= 1000