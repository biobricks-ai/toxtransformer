# PYTHONPATH=./ spark-submit --master local[*] --driver-memory 512g --conf spark.driver.maxResultSize=256g --conf spark.memory.offHeap.enabled=true --conf spark.memory.offHeap.size=256g code/5_0_1_consolidate_evaluations.py 2> cache/consolidate_evaluations/logs/err.log

import pathlib
import logging
import shutil
import sklearn.metrics
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Set up directories
outdir = pathlib.Path("cache/consolidate_evaluations")
evalsdir = pathlib.Path("cache/generate_evaluations/evaluations.parquet")

# Remove old output
partitioned_dir = outdir / "multitask_predictions.parquet"
shutil.rmtree(partitioned_dir, ignore_errors=True)
partitioned_dir.mkdir(parents=True, exist_ok=True)

# Set up logging
outdir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=outdir / "logs" / "consolidate_evaluations.log",
    filemode="w"
)
logging.info("Starting consolidate_evaluations with Spark.")

# Start Spark session with increased memory settings and custom temp directory
# .config("spark.local.dir", "/data/toxtransformer/spark-temp") \
spark = SparkSession.builder \
    .appName("ConsolidateEvaluations") \
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

# Create temp directory on the larger filesystem
# spark_temp_dir = pathlib.Path("/data/toxtransformer/spark-temp")
# spark_temp_dir.mkdir(parents=True, exist_ok=True)

# Log Spark UI URLs for monitoring
ui_web_url = spark.sparkContext.uiWebUrl
logging.info(f"Spark UI available at: {ui_web_url}")

# Load parquet directory
df = spark.read.parquet(str(evalsdir))
logging.info(f"{len(list(evalsdir.glob('*.parquet')))} Parquet files loaded.")

# Write output with smaller partitions to reduce memory pressure
df.write \
    .mode("overwrite") \
    .option("compression", "zstd") \
    .option("maxRecordsPerFile", 5_000_000) \
    .parquet(str(partitioned_dir))

logging.info(f"Partitioned dataset written to {partitioned_dir}")