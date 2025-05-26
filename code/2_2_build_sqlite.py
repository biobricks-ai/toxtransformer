# PYTHONPATH=./ spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///tmp/spark-events code/2_build_sqlite.py 2> cache/build_sqlite/err.log
import os, sys, biobricks as bb, pandas as pd, shutil, sqlite3, pathlib
import pyspark.sql, pyspark.sql.functions as F
import pyarrow.dataset as ds
from tqdm import tqdm
import logging
import cvae.tokenizer.selfies_property_val_tokenizer as spt

logging.basicConfig(level=logging.INFO, filename='cache/build_sqlite/build_sqlite.log', filemode='w')
logger = logging.getLogger(__name__)

#%% SETUP =================================================================================
logger.info("Initializing Spark session")
spark = pyspark.sql.SparkSession.builder.appName("ChemharmonyDataProcessing")
spark = spark.config("spark.driver.memory", "64g").config("spark.driver.maxResultSize", "100g").getOrCreate()
    
ch = bb.assets('chemharmony')
outdir = pathlib.Path('cache/build_sqlite')
outdir.mkdir(parents=True, exist_ok=True)

logger.info("Loading tokenizer")
tokenizer = spt.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
broadcast_tokenizer = spark.sparkContext.broadcast(tokenizer)

#%% BUILD PROPERTY TABLES =================================================================
logger.info("Building property tables")
pytorch_id_to_property_token = lambda x : broadcast_tokenizer.value.assay_id_to_token_idx(int(x))
pytorch_id_to_property_token_udf = F.udf(pytorch_id_to_property_token, pyspark.sql.types.LongType())

binval_to_value_token = lambda x : broadcast_tokenizer.value.value_id_to_token_idx(int(x))
binval_to_value_token_udf = F.udf(binval_to_value_token, pyspark.sql.types.LongType())

raw_activities = spark.read.parquet("cache/preprocess_activities/activities.parquet")\
    .withColumnRenamed('assay','property_id')\
    .withColumnRenamed('sid','substance_id')\
    .withColumn("property_token", pytorch_id_to_property_token_udf("assay_index"))\
    .withColumn('value_token',binval_to_value_token_udf('value'))\
    .filter(F.col("property_token").isNotNull())\
    .select("property_id","source","property_token",'substance_id','smiles','selfies','value','value_token')

raw_property_tokens = raw_activities.select('property_id','property_token').distinct()
raw_prop_title = spark.read.parquet(ch.property_titles_parquet).withColumnRenamed('pid', 'property_id')

prop = spark.read.parquet(ch.properties_parquet)
prop = prop.withColumnRenamed('pid', 'property_id')
prop = raw_property_tokens.join(prop, on='property_id', how='left').join(raw_prop_title, on='property_id', how='left').cache()

raw_prop_cat = spark.read.parquet(ch.property_categories_parquet)
raw_prop_cat = raw_prop_cat.withColumnRenamed('pid', 'property_id').cache()

## categories and property_category
logger.info("Processing categories and property categories")
cat = raw_prop_cat.select('category').distinct()
cat = cat.withColumn('category_id', F.monotonically_increasing_id())
prop_cat = raw_prop_cat.join(cat, on='category').select('property_id', 'category_id','reason','strength')

## sources and property_source
src = prop.select('source').distinct()
src = src.withColumn('source_id', F.monotonically_increasing_id())
prop = prop.join(src, on='source').select('property_id','title','property_token','source_id','data')

## substances
substances = spark.read.parquet("cache/preprocess/substances2.parquet").select('sid','inchi').distinct()
substances = substances.withColumnRenamed('sid','substance_id')

## activities and activity_source 
activities = raw_activities\
    .join(src, on='source')\
    .join(substances, on='substance_id')\
    .select('source_id','property_id','property_token','substance_id','inchi','smiles','value','value_token')

property_summary_statistics = raw_activities.groupBy('property_id')\
    .agg(
        F.sum(F.when(F.col('value') == 1, 1).otherwise(0)).alias('positive_count'),
        F.sum(F.when(F.col('value') == 0, 1).otherwise(0)).alias('negative_count')
    )


# WRITE LARGE TABLES TO SQLITE =============================================================
def parquet_to_sqlite(parquet_path, table_name):
    logger.info(f"Writing {table_name} to SQLite")
    conn = sqlite3.connect((outdir / 'cvae.sqlite').as_posix())
    
    import pyarrow.dataset as ds
    dataset = ds.dataset(parquet_path, format="parquet")
    batches = dataset.to_batches()
    
    # Get table name and schema from first batch
    first_batch = next(batches)
    
    # Create table
    first_batch.to_pandas().head(0).to_sql(table_name, conn, if_exists='replace', index=False)
    
    # Write batches with progress bar
    cursor = conn.cursor()
    total_rows = sum(batch.num_rows for batch in dataset.to_batches())
    with tqdm(total=total_rows, desc=f"Writing {table_name}") as pbar:
        for batch in dataset.to_batches():
            df = batch.to_pandas()
            placeholders = ','.join(['?' for _ in df.columns])
            insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
            cursor.executemany(insert_sql, df.values.tolist())
            pbar.update(batch.num_rows)

    conn.commit()
    conn.close()

# remove the old sqlite file if it exists
if (outdir / 'cvae.sqlite').exists():
    os.remove((outdir / 'cvae.sqlite').as_posix())

tmpdir = outdir / 'tmp'
tmpdir.mkdir(exist_ok=True)
tables = [prop, cat, prop_cat, src, activities, property_summary_statistics]
tablename = ['property', 'category', 'property_category', 'source', 'activity', 'property_summary_statistics']
for table, name in zip(tables, tablename):
    logger.info(f"Creating {name} table")
    table.write.parquet((tmpdir / f'{name}.parquet').as_posix(), mode='overwrite')
    parquet_to_sqlite((tmpdir / f'{name}.parquet').as_posix(), name)
    # test that there are at least 3 rows in the table
    with sqlite3.connect((outdir / 'cvae.sqlite').as_posix()) as conn:
        assert pd.read_sql_query(f"SELECT COUNT(*) FROM {name}", conn).iloc[0]['COUNT(*)'] >= 3, f"Table {name} has less than 3 rows"

shutil.rmtree(tmpdir)

## CREATE INDEXES =============================================================
logger.info("Creating SQLite indexes")
conn = sqlite3.connect((outdir / 'cvae.sqlite').as_posix())
cursor = conn.cursor()

# Create indexes
cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_source_id ON activity (source_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_property_id ON activity (property_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_source_source_id ON source (source_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_property_id ON property (property_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_category_property_id ON property_category (property_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_category_category_id ON category (category_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_property_category_category_id ON property_category (category_id);")
cursor.execute("CREATE INDEX IF NOT EXISTS idx_inchi ON activity (inchi);")

conn.commit()
conn.close()

# MOVE RESULT TO BRICK/cvae.sqlite =============================================================
logger.info("Moving SQLite database to final location")
shutil.copy((outdir / 'cvae.sqlite').as_posix(), 'brick/cvae.sqlite')

# DO A SIMPLE TEST QUERY =============================================================
logger.info("Running test queries")
conn = sqlite3.connect('brick/cvae.sqlite')

query = """
SELECT * 
FROM property pr 
INNER JOIN property_category pc ON pr.property_id = pc.property_id
INNER JOIN category c ON pc.category_id = c.category_id
WHERE c.category = 'endocrine disruption' 
ORDER BY strength DESC
"""

df = pd.read_sql_query(query, conn)

assert df['data'].isnull().sum() == 0, "Null values found in 'data' column"
assert df['reason'].isnull().sum() == 0, "Null values found in 'reason' column"

assert pd.api.types.is_string_dtype(df['data']), "'data' column should be of type string"
assert pd.api.types.is_string_dtype(df['reason']), "'reason' column should be of type string"
# assert pd.api.types.is_numeric_dtype(df['strength']), "'strength' column should be of type float"

conn.close()
logger.info("Database creation and validation completed successfully")