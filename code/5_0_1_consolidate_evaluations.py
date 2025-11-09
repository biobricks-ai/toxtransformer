"""
rm -rf cache/consolidate_evaluations/
for SPLIT in {0..5}; do
  INDIR="cache/generate_evaluations/${SPLIT}/evaluations.parquet"
  OUTDIR="cache/consolidate_evaluations/${SPLIT}"
  mkdir -p "$OUTDIR/logs"
  
  SPLIT=$SPLIT INDIR="$INDIR" OUTDIR="$OUTDIR" PYTHONPATH=./ \
  spark-submit --master local[*] \
    --driver-memory 512g \
    --conf spark.driver.maxResultSize=256g \
    --conf spark.memory.offHeap.enabled=true \
    --conf spark.memory.offHeap.size=256g \
    code/5_0_1_consolidate_evaluations.py \
    2> "$OUTDIR/logs/err.log"
done
"""

import os
import pathlib
import logging
import shutil
import sklearn.metrics
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Get environment variables
SPLIT = os.getenv("SPLIT")
INDIR = os.getenv("INDIR")
OUTDIR = os.getenv("OUTDIR")

# Set up directories
outdir = pathlib.Path(OUTDIR)
evalsdir = pathlib.Path(INDIR)

# Remove old output
partitioned_dir = outdir / "multitask_predictions.parquet"
shutil.rmtree(partitioned_dir, ignore_errors=True)
partitioned_dir.mkdir(parents=True, exist_ok=True)

# Set up logging
logdir = outdir / "logs"
logdir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=logdir / "consolidate_evaluations.log",
    filemode="w"
)
logging.info(f"Starting consolidate_evaluations for SPLIT={SPLIT} with Spark.")

# Start Spark session with increased memory settings
spark = SparkSession.builder \
    .appName(f"ConsolidateEvaluations_Split{SPLIT}") \
    .config("spark.sql.parquet.compression.codec", "zstd") \
    .config("spark.driver.memory", "512g") \
    .config("spark.driver.maxResultSize", "256g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "256g") \
    .config("spark.sql.shuffle.partitions", "400") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.sql.adaptive.skewJoin.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .getOrCreate()

# Log Spark UI URLs for monitoring
ui_web_url = spark.sparkContext.uiWebUrl
logging.info(f"Spark UI available at: {ui_web_url}")

# Load parquet directory
logging.info(f"Reading from {evalsdir}")
df = spark.read.parquet(str(evalsdir))
parquet_files = list(evalsdir.glob('*.parquet'))
logging.info(f"{len(parquet_files)} Parquet files loaded from split {SPLIT}.")
logging.info(f"Total records before deduplication: {df.count()}")

# Drop duplicates across all columns
df = df.dropDuplicates()
logging.info(f"Total records after deduplication: {df.count()}")

# Write output with smaller partitions to reduce memory pressure
df.write \
    .mode("overwrite") \
    .option("compression", "zstd") \
    .option("maxRecordsPerFile", 5_000_000) \
    .parquet(str(partitioned_dir))

logging.info(f"Partitioned dataset written to {partitioned_dir}")

# Stop Spark session
spark.stop()
logging.info(f"Consolidation complete for SPLIT={SPLIT}")