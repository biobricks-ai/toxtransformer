# CMD
"""
rm -rf cache/build_sqlite
mkdir -p cache/build_sqlite
PYTHONPATH=./ spark-submit --master "local[*]" \
--driver-memory 218g --conf spark.eventLog.enabled=true \
--conf spark.eventLog.dir=file:///tmp/spark-events \
code/2_2_0_build_sqlite.py 2> cache/build_sqlite/err.log
"""

import os, biobricks as bb, pandas as pd, shutil, sqlite3, pathlib
import pyspark.sql, pyspark.sql.functions as F
import logging
import torch
import sys
import time
from datetime import datetime

# Setup output directory first
outdir = pathlib.Path('cache/build_sqlite')
outdir.mkdir(parents=True, exist_ok=True)

# Setup logging to both file and stderr with timestamps
log_file = outdir / 'progress.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

def log_progress(msg):
    """Log with immediate flush to ensure visibility."""
    logger.info(msg)
    # Force flush handlers
    for handler in logger.handlers:
        handler.flush()

log_progress("="*60)
log_progress("BUILD SQLITE DATABASE")
log_progress("="*60)

#region SETUP =================================================================================
log_progress("Initializing Spark session")
spark = pyspark.sql.SparkSession.builder.appName("ChemharmonyDataProcessing")
spark = spark.config("spark.driver.maxResultSize", "100g").config("spark.local.dir", "/tmp/spark-local").getOrCreate()

ch = bb.assets('chemharmony')
# endregion

#region BUILD TABLES ==========================================================================
log_progress("Building Spark tables...")

raw_activities = spark.read.parquet("cache/preprocess_activities/activities_augmented.parquet")\
    .withColumnRenamed('assay','property_id')\
    .withColumnRenamed('sid','substance_id')\
    .withColumn("property_token", F.col("assay_index"))\
    .withColumn('selfies_tokens', F.array_join(F.col('encoded_selfies'), ' '))\
    .filter(F.col("property_token").isNotNull())\
    .select("property_id","source","property_token",'substance_id','smiles','selfies','selfies_tokens','value')

raw_property_tokens = raw_activities.select('property_id','property_token').distinct()
raw_prop_title = spark.read.parquet(ch.property_titles_parquet).withColumnRenamed('pid', 'property_id')
raw_prop_cat = spark.read.parquet(ch.property_categories_parquet).withColumnRenamed('pid', 'property_id').cache()

proptable = spark.read.parquet(ch.properties_parquet).withColumnRenamed('pid', 'property_id')
proptable = raw_property_tokens.join(proptable, on='property_id', how='left').join(raw_prop_title, on='property_id', how='left').cache()

cat = raw_prop_cat.select('category').distinct().withColumn('category_id', F.monotonically_increasing_id())
prop_cat = raw_prop_cat.join(cat, on='category').select('property_id', 'category_id','reason','strength')

src = proptable.select('source').distinct().withColumn('source_id', F.monotonically_increasing_id())
proptable = proptable.join(src, on='source').select('property_id','title','property_token','source_id','data')

substances = spark.read.parquet("cache/preprocess/substances2.parquet")\
    .select('sid','inchi').distinct().withColumnRenamed('sid','substance_id')

activities = raw_activities.join(src, on='source').join(substances, on='substance_id')\
    .select('source_id','property_id','property_token','substance_id','inchi','smiles','selfies','selfies_tokens','value')

property_summary_statistics = raw_activities.groupBy('property_id').agg(
    F.sum(F.when(F.col('value') == 1, 1).otherwise(0)).alias('positive_count'),
    F.sum(F.when(F.col('value') == 0, 1).otherwise(0)).alias('negative_count')
)
log_progress("Spark tables built")
# endregion

#region BUILD HOLDOUT TABLE ===================================================================
log_progress("Building holdout table from bootstrap splits")

holdout_records = []
N_BOOTSTRAP = 5

for split_id in range(N_BOOTSTRAP):
    log_progress(f"Processing bootstrap split {split_id}")
    test_dir = pathlib.Path(f"cache/build_tensordataset/bootstrap/split_{split_id}/test")

    tensor_files = list(test_dir.glob("*.pt"))
    for i, tensor_file in enumerate(tensor_files):
        data = torch.load(tensor_file, map_location="cpu", weights_only=True)

        for selfies_enc, properties, values in zip(data["selfies"], data["properties"], data["values"]):
            valid_mask = (properties != -1) & (values != -1)
            if not valid_mask.any():
                continue

            selfies_tokens = ' '.join(str(x) for x in selfies_enc.tolist())
            for prop, val in zip(properties[valid_mask].tolist(), values[valid_mask].tolist()):
                holdout_records.append({
                    'split_id': split_id,
                    'selfies_tokens': selfies_tokens,
                    'property_token': prop,
                    'value': val
                })

        if (i + 1) % 10 == 0:
            log_progress(f"  Split {split_id}: processed {i+1}/{len(tensor_files)} files, {len(holdout_records)} records so far")

log_progress(f"Collected {len(holdout_records)} holdout records across {N_BOOTSTRAP} splits")
holdout_df = spark.createDataFrame(pd.DataFrame(holdout_records))
# endregion

#region WRITE TO SQLITE =======================================================================
def parquet_to_sqlite_chunked(parquet_path, table_name, chunk_size=100000, commit_every=1000000):
    """
    Write parquet to SQLite with streaming reads and regular progress logging.

    Args:
        parquet_path: Path to parquet file/directory
        table_name: SQLite table name
        chunk_size: Rows per batch read
        commit_every: Commit transaction after this many rows
    """
    import pyarrow.dataset as ds

    log_progress(f"[{table_name}] Starting SQLite write from {parquet_path}")

    dataset = ds.dataset(parquet_path, format="parquet")

    # Count total rows first (streaming count)
    log_progress(f"[{table_name}] Counting rows...")
    total_rows = 0
    for batch in dataset.to_batches(batch_size=chunk_size):
        total_rows += batch.num_rows
    log_progress(f"[{table_name}] Total rows: {total_rows:,}")

    conn = sqlite3.connect((outdir / 'cvae.sqlite').as_posix())

    # Create table from first batch schema
    first_batch = next(dataset.to_batches(batch_size=1))
    first_batch.to_pandas().head(0).to_sql(table_name, conn, if_exists='replace', index=False)

    cursor = conn.cursor()
    rows_written = 0
    rows_since_commit = 0
    start_time = time.time()
    last_log_time = start_time

    log_progress(f"[{table_name}] Writing rows...")

    for batch in dataset.to_batches(batch_size=chunk_size):
        df = batch.to_pandas()
        placeholders = ','.join(['?' for _ in df.columns])
        cursor.executemany(f"INSERT INTO {table_name} VALUES ({placeholders})", df.values.tolist())

        rows_written += batch.num_rows
        rows_since_commit += batch.num_rows

        # Commit periodically to avoid huge transactions
        if rows_since_commit >= commit_every:
            conn.commit()
            rows_since_commit = 0

        # Log progress every 10 seconds or every 1M rows
        current_time = time.time()
        if current_time - last_log_time >= 10 or rows_written % 1000000 < chunk_size:
            elapsed = current_time - start_time
            rate = rows_written / elapsed if elapsed > 0 else 0
            pct = 100.0 * rows_written / total_rows if total_rows > 0 else 0
            eta = (total_rows - rows_written) / rate if rate > 0 else 0
            log_progress(f"[{table_name}] {rows_written:,}/{total_rows:,} ({pct:.1f}%) - {rate:,.0f} rows/sec - ETA: {eta/60:.1f} min")
            last_log_time = current_time

    # Final commit
    conn.commit()
    conn.close()

    elapsed = time.time() - start_time
    log_progress(f"[{table_name}] COMPLETE: {rows_written:,} rows in {elapsed/60:.1f} minutes ({rows_written/elapsed:,.0f} rows/sec)")

# Remove old sqlite files
log_progress("Removing old SQLite files...")
for path in [(outdir / 'cvae.sqlite'), pathlib.Path('brick/cvae.sqlite')]:
    if path.exists():
        os.remove(path.as_posix())
        log_progress(f"  Removed {path}")

# Write all tables
tmpdir = outdir / 'tmp'
tmpdir.mkdir(exist_ok=True)

tables = [proptable, cat, prop_cat, src, activities, property_summary_statistics, holdout_df]
tablenames = ['property', 'category', 'property_category', 'source', 'activity', 'property_summary_statistics', 'holdout_samples']

for table, name in zip(tables, tablenames):
    log_progress(f"="*40)
    log_progress(f"Processing table: {name}")
    log_progress(f"="*40)

    # Write parquet with repartitioning for large tables
    parquet_path = (tmpdir / f'{name}.parquet').as_posix()

    if name == 'activity':
        # Large table - repartition for parallel write
        log_progress(f"[{name}] Writing parquet (repartitioned to 200 files)...")
        table.repartition(200).write.parquet(parquet_path, mode='overwrite')
    else:
        log_progress(f"[{name}] Writing parquet...")
        table.write.parquet(parquet_path, mode='overwrite')

    log_progress(f"[{name}] Parquet write complete")

    # Write to SQLite
    parquet_to_sqlite_chunked(parquet_path, name)

    # Verify
    with sqlite3.connect((outdir / 'cvae.sqlite').as_posix()) as conn:
        count = pd.read_sql_query(f"SELECT COUNT(*) as cnt FROM {name}", conn).iloc[0]['cnt']
        log_progress(f"[{name}] Verified: {count:,} rows in SQLite")
        assert count >= 3, f"Table {name} has less than 3 rows"

log_progress("Cleaning up temp parquet files...")
shutil.rmtree(tmpdir)
# endregion

#region CREATE INDEXES ========================================================================
log_progress("="*40)
log_progress("Creating indexes")
log_progress("="*40)

with sqlite3.connect((outdir / 'cvae.sqlite').as_posix()) as conn:
    cursor = conn.cursor()

    indexes = [
        ("idx_activity_source_id", "activity (source_id)"),
        ("idx_activity_property_id", "activity (property_id)"),
        ("idx_activity_inchi", "activity (inchi)"),
        ("idx_activity_selfies", "activity (selfies)"),
        ("idx_activity_selfies_tokens", "activity (selfies_tokens)"),
        ("idx_activity_selfies_property", "activity (selfies, property_token)"),
        ("idx_source_source_id", "source (source_id)"),
        ("idx_property_property_id", "property (property_id)"),
        ("idx_property_category_property_id", "property_category (property_id)"),
        ("idx_category_category_id", "category (category_id)"),
        ("idx_property_category_category_id", "property_category (category_id)"),
        ("idx_holdout_split_id", "holdout_samples (split_id)"),
        ("idx_holdout_selfies_tokens", "holdout_samples (selfies_tokens)"),
        ("idx_holdout_property_token", "holdout_samples (property_token)"),
        ("idx_holdout_split_selfies_tokens", "holdout_samples (split_id, selfies_tokens)"),
    ]

    for idx_name, idx_def in indexes:
        log_progress(f"Creating index: {idx_name}")
        cursor.execute(f"CREATE INDEX {idx_name} ON {idx_def};")

    conn.commit()
    log_progress("All indexes created")
# endregion

#region FINALIZE ==============================================================================
log_progress("="*40)
log_progress("Finalizing")
log_progress("="*40)

log_progress("Copying to brick/cvae.sqlite...")
shutil.copy((outdir / 'cvae.sqlite').as_posix(), 'brick/cvae.sqlite')
log_progress("Copy complete")

# Validation
log_progress("Running validation query...")
conn = sqlite3.connect('brick/cvae.sqlite')
df = pd.read_sql_query("""
    SELECT * FROM property pr
    INNER JOIN property_category pc ON pr.property_id = pc.property_id
    INNER JOIN category c ON pc.category_id = c.category_id
    WHERE c.category = 'endocrine disruption'
    ORDER BY strength DESC
""", conn)
assert df['data'].isnull().sum() == 0 and df['reason'].isnull().sum() == 0
log_progress(f"Validation passed: {len(df)} endocrine disruption properties")
conn.close()

# Final stats
final_size = os.path.getsize('brick/cvae.sqlite') / (1024**3)
log_progress(f"Final database size: {final_size:.1f} GB")

log_progress("="*60)
log_progress("BUILD SQLITE COMPLETE!")
log_progress("="*60)
# endregion
