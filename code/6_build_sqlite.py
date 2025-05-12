# spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///tmp/spark-events code/6_build_sqlite.py

import os, sys, biobricks as bb, pandas as pd, shutil, sqlite3, pathlib
import pyspark.sql, pyspark.sql.functions as F

sys.path.insert(0, os.getcwd())
import cvae.tokenizer.selfies_property_val_tokenizer as spt

#%% SETUP =================================================================================
spark = pyspark.sql.SparkSession.builder.appName("ChemharmonyDataProcessing")
spark = spark.config("spark.driver.memory", "64g").config("spark.driver.maxResultSize", "100g").getOrCreate()
    
ch = bb.assets('chemharmony')
outdir = pathlib.Path('cache/build_sqlite')
outdir.mkdir(parents=True, exist_ok=True)

tokenizer = spt.SelfiesPropertyValTokenizer.load('brick/moe/spvt_tokenizer')

#%% BUILD PROPERTY TABLES =================================================================
pytorch_id_to_property_token = lambda x : tokenizer.assay_id_to_token_idx(x)
pytorch_id_to_property_token_udf = F.udf(pytorch_id_to_property_token, pyspark.sql.types.LongType())
property_tokens = spark.read.parquet("cache/preprocess_activities/activities.parquet")\
    .withColumnRenamed('assay','property_id')\
    .withColumnRenamed('assay_index','property_pytorch_index')\
    .withColumn("property_pytorch_index", F.col("property_pytorch_index").cast("int"))\
    .withColumn("property_token", pytorch_id_to_property_token_udf("property_pytorch_index"))\
    .select("property_id","property_token").distinct().cache()

binval_to_value_token = lambda x : tokenizer.value_id_to_token_idx(int(x))
binval_to_value_token_udf = F.udf(binval_to_value_token, pyspark.sql.types.LongType())
raw_activities = spark.read.parquet(ch.activities_parquet)\
    .withColumnRenamed('sid', 'substance_id')\
    .withColumnRenamed('aid', 'activity_id')\
    .withColumnRenamed('pid', 'property_id')\
    .join(property_tokens, on='property_id')\
    .withColumn('value_token',binval_to_value_token_udf('binary_value'))\
    .select('source','activity_id','property_id','property_token','substance_id','inchi','smiles','value','binary_value','value_token')\
    .cache()

raw_prop_title = spark.read.parquet(ch.property_titles_parquet).withColumnRenamed('pid', 'property_id')

prop = spark.read.parquet(ch.properties_parquet)
prop = prop.withColumnRenamed('pid', 'property_id')
prop = property_tokens.join(prop, on='property_id', how='left').join(raw_prop_title, on='property_id', how='left').cache()

raw_prop_cat = spark.read.parquet(ch.property_categories_parquet)
raw_prop_cat = raw_prop_cat.withColumnRenamed('pid', 'property_id').cache()

## categories and property_category
cat = raw_prop_cat.select('category').distinct()
cat = cat.withColumn('category_id', F.monotonically_increasing_id())
prop_cat = raw_prop_cat.join(cat, on='category').select('property_id', 'category_id','reason','strength')

## sources and property_source
src = prop.select('source').distinct()
src = src.withColumn('source_id', F.monotonically_increasing_id())
prop = prop.join(src, on='source').select('property_id','title','property_token','source_id','data')

## activities and activity_source 
activities = raw_activities\
    .join(src, on='source')\
    .select('source_id','activity_id','property_id','property_token','substance_id','inchi','smiles','value','binary_value','value_token')\
    .cache()

# WRITE TABLE TO SQLITE =============================================================
conn = sqlite3.connect((outdir / 'cvae.sqlite').as_posix())

prop.toPandas().to_sql('property', conn, if_exists='replace', index=False)

evaldf = spark.read.parquet('cache/eval_multi_properties/multitask_metrics.parquet')
evaldf = evaldf.withColumnRenamed('assay', 'property_token').cache()

# assert that all property_tokens in evaldf are in property_tokens
# Check if all property tokens in evaldf are in property_tokens
evaldf_tokens_not_in_property_tokens = evaldf.join(
    prop.select("property_token"),
    on="property_token",
    how="left_anti"
).count()
assert evaldf_tokens_not_in_property_tokens == 0, "Some property tokens in evaldf are not in prop"


cat.toPandas().to_sql('category', conn, if_exists='replace', index=False)
prop_cat.toPandas().to_sql('property_category', conn, if_exists='replace', index=False)

src.toPandas().to_sql('source', conn, if_exists='replace', index=False)
activities.toPandas().to_sql('activity', conn, if_exists='replace', index=False)

conn.close()

## CREATE INDEXES =============================================================
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
shutil.move((outdir / 'cvae.sqlite').as_posix(), 'brick/cvae.sqlite')

# DO A SIMPLE TEST QUERY =============================================================
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
assert pd.api.types.is_numeric_dtype(df['strength']), "'reason' column should be of type string"

conn.close()