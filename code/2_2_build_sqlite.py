# CMD
"""
rm -r cache/build_sqlite
mkdir -p cache/build_sqlite
PYTHONPATH=./ spark-submit --master "local[*]" \
--driver-memory 218g --conf spark.eventLog.enabled=true \
--conf spark.eventLog.dir=file:///tmp/spark-events \
code/2_2_build_sqlite.py 2> cache/build_sqlite/err.log; \
./slackmsg 'Build sqlite finished'
"""
# PYTHONPATH=./ spark-submit --master "local[*]" --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///tmp/spark-events code/2_2_build_sqlite.py 2> cache/build_sqlite/err.log
# PYTHONPATH=./ spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///tmp/spark-events --conf spark.local.dir=/tmp/spark-local code/2_2_build_sqlite.py 2> cache/build_sqlite/err.log; ./slackmsg 'Build sqlite finished'

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
spark = spark.config("spark.driver.maxResultSize", "100g").config("spark.local.dir", "/tmp/spark-local").getOrCreate()

ch = bb.assets('chemharmony')
outdir = pathlib.Path('cache/build_sqlite')
outdir.mkdir(parents=True, exist_ok=True)

#%% BUILD PROPERTY TABLES =================================================================
logger.info("Building property tables")

raw_activities = spark.read.parquet("cache/preprocess_activities/activities_augmented.parquet")\
    .withColumnRenamed('assay','property_id')\
    .withColumnRenamed('sid','substance_id')\
    .withColumn("property_token", F.col("assay_index"))\
    .withColumn('value_token', F.col('value'))\
    .filter(F.col("property_token").isNotNull())\
    .select("property_id","source","property_token",'substance_id','smiles','selfies','value','value_token')

raw_property_tokens = raw_activities.select('property_id','property_token').distinct()

raw_prop_title = spark.read.parquet(ch.property_titles_parquet).withColumnRenamed('pid', 'property_id')

proptable = spark.read.parquet(ch.properties_parquet)
proptable = proptable.withColumnRenamed('pid', 'property_id')
proptable = raw_property_tokens.join(proptable, on='property_id', how='left').join(raw_prop_title, on='property_id', how='left').cache()

raw_prop_cat = spark.read.parquet(ch.property_categories_parquet)
raw_prop_cat = raw_prop_cat.withColumnRenamed('pid', 'property_id').cache()

## categories and property_category
logger.info("Processing categories and property categories")
cat = raw_prop_cat.select('category').distinct()
cat = cat.withColumn('category_id', F.monotonically_increasing_id())
prop_cat = raw_prop_cat.join(cat, on='category').select('property_id', 'category_id','reason','strength')

## sources and property_source
src = proptable.select('source').distinct()
src = src.withColumn('source_id', F.monotonically_increasing_id())
proptable = proptable.join(src, on='source').select('property_id','title','property_token','source_id','data')

## substances
substances = spark.read.parquet("cache/preprocess/substances2.parquet").select('sid','inchi').distinct()
substances = substances.withColumnRenamed('sid','substance_id')

## activities and activity_source
activities = raw_activities\
    .join(src, on='source')\
    .join(substances, on='substance_id')\
    .select('source_id','property_id','property_token','substance_id','inchi','smiles','selfies','value','value_token')

property_summary_statistics = raw_activities.groupBy('property_id')\
    .agg(
        F.sum(F.when(F.col('value') == 1, 1).otherwise(0)).alias('positive_count'),
        F.sum(F.when(F.col('value') == 0, 1).otherwise(0)).alias('negative_count')
    )

#%% BUILD HOLDOUT TRACKING TABLE =============================================================
logger.info("Building holdout tracking table from bootstrap splits")

# We need to extract (selfies, property_token) pairs from each bootstrap test split
# The tensor files were built from activities_augmented.parquet grouped by (smiles, encoded_selfies, assay_index)
# We'll rebuild this mapping by re-reading activities_augmented and using smiles->selfies mapping

logger.info("Building smiles to selfies lookup from activities")
# Get unique smiles -> selfies mapping
smiles_to_selfies = spark.read.parquet("cache/preprocess_activities/activities_augmented.parquet")\
    .select('smiles', 'selfies', 'encoded_selfies').distinct()

# Convert to pandas for fast lookup
smiles_lookup_df = smiles_to_selfies.toPandas()
logger.info(f"Loaded {len(smiles_lookup_df)} unique smiles->selfies mappings")

# Create lookup: tuple(encoded_selfies with padding) -> (smiles, selfies)
# Both tensor files and parquet should be padded to 120 tokens
encoding_to_info = {}
for _, row in smiles_lookup_df.iterrows():
    # Use full encoded_selfies including padding (should be 120 tokens)
    key = tuple(row['encoded_selfies'])
    encoding_to_info[key] = (row['smiles'], row['selfies'])

logger.info(f"Created lookup with {len(encoding_to_info)} entries")

# Process each bootstrap split
import torch
holdout_records = []

N_BOOTSTRAP = 5
for split_id in range(N_BOOTSTRAP):
    logger.info(f"Processing bootstrap split {split_id}")
    test_dir = pathlib.Path(f"cache/build_tensordataset/bootstrap/split_{split_id}/test")

    if not test_dir.exists():
        logger.warning(f"Test directory not found for split {split_id}: {test_dir}")
        continue

    # Load all tensor files for this split
    for tensor_file in test_dir.glob("*.pt"):
        data = torch.load(tensor_file, map_location="cpu", weights_only=True)

        selfies_tensor = data["selfies"]  # [N, seq_len] - encoded selfies
        properties_tensor = data["properties"]  # [N, max_props]
        values_tensor = data["values"]  # [N, max_props]

        # Process each sample
        for encoded_selfies, properties, values in zip(selfies_tensor, properties_tensor, values_tensor):
            # Filter out padding from properties/values
            valid_mask = (properties != -1) & (values != -1)
            valid_properties = properties[valid_mask]
            valid_values = values[valid_mask]

            if len(valid_properties) == 0:
                continue

            # Look up selfies string from encoded version (use full 120-length padded version)
            encoded_key = tuple(encoded_selfies.tolist())
            info = encoding_to_info.get(encoded_key)

            if info is None:
                # Try logging first few to debug
                if len(holdout_records) < 5:
                    logger.warning(f"Could not find mapping for encoded key length {len(encoded_key)}, first 10: {encoded_key[:10]}")
                continue

            smiles_str, selfies_str = info

            # Add a record for each property in this sample
            for prop, val in zip(valid_properties.tolist(), valid_values.tolist()):
                holdout_records.append({
                    'split_id': split_id,
                    'selfies': selfies_str,
                    'property_token': prop,  # Raw assay_index
                    'value': val
                })

logger.info(f"Collected {len(holdout_records)} holdout records across {N_BOOTSTRAP} splits")

# Convert to Spark DataFrame
holdout_df = spark.createDataFrame(pd.DataFrame(holdout_records))
holdout_count = holdout_df.count()
logger.info(f"Created holdout DataFrame with {holdout_count} rows")


# WRITE LARGE TABLES TO SQLITE =============================================================
def parquet_to_sqlite(parquet_path, table_name):
    logger.info(f"Writing {table_name} to SQLite")
    with sqlite3.connect((outdir / 'cvae.sqlite').as_posix()) as conn:
        
        import pyarrow.dataset as ds
        dataset = ds.dataset(parquet_path, format="parquet")
        
        # Convert to list to avoid iterator consumption issues
        batches = list(dataset.to_batches())
        
        # Get table name and schema from first batch
        first_batch = batches[0]
        
        # Create table
        first_batch.to_pandas().head(0).to_sql(table_name, conn, if_exists='replace', index=False)
        
        # Write batches with progress bar
        cursor = conn.cursor()
        total_rows = sum(batch.num_rows for batch in batches)
        with tqdm(total=total_rows, desc=f"Writing {table_name}") as pbar:
            for batch in batches:
                df = batch.to_pandas()
                placeholders = ','.join(['?' for _ in df.columns])
                insert_sql = f"INSERT INTO {table_name} VALUES ({placeholders})"
                cursor.executemany(insert_sql, df.values.tolist())
                pbar.update(batch.num_rows)

        conn.commit()

# remove the old sqlite files if they exist
if (outdir / 'cvae.sqlite').exists():
    logger.info("Removing old cache/build_sqlite/cvae.sqlite")
    os.remove((outdir / 'cvae.sqlite').as_posix())
if pathlib.Path('brick/cvae.sqlite').exists():
    logger.info("Removing old brick/cvae.sqlite")
    os.remove('brick/cvae.sqlite')

tmpdir = outdir / 'tmp'
tmpdir.mkdir(exist_ok=True)
tables = [proptable, cat, prop_cat, src, activities, property_summary_statistics, holdout_df]
tablename = ['property', 'category', 'property_category', 'source', 'activity', 'property_summary_statistics', 'holdout_samples']
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
with sqlite3.connect((outdir / 'cvae.sqlite').as_posix()) as conn:
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
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_selfies ON activity (selfies);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_activity_selfies_property ON activity (selfies, property_token);")

    # Create indexes for holdout_samples table
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_holdout_split_id ON holdout_samples (split_id);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_holdout_selfies ON holdout_samples (selfies);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_holdout_property_token ON holdout_samples (property_token);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_holdout_split_selfies ON holdout_samples (split_id, selfies);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_holdout_split_property ON holdout_samples (split_id, property_token);")

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