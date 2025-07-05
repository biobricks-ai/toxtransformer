# PYTHONPATH=./ spark-submit --master local[*] --driver-memory 512g --conf spark.driver.maxResultSize=256g --conf spark.memory.offHeap.enabled=true --conf spark.memory.offHeap.size=256g code/5_0_1_consolidate_evaluations.py

import pathlib
import logging
import shutil
import sklearn.metrics
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

# Set up directories
outdir = pathlib.Path("cache/consolidate_evaluations")
tmpdir = pathlib.Path("cache/generate_evaluations/evaluations.parquet")
tmpdir.mkdir(parents=True, exist_ok=True)
partitioned_dir = outdir / "multitask_predictions.parquet"

# Set up logging
outdir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=outdir / "consolidate_evaluations.log",
    filemode="w"
)
logging.info("Starting consolidate_evaluations with Spark.")

# Start Spark session with increased memory settings
spark = SparkSession.builder \
    .appName("ConsolidateEvaluations") \
    .config("spark.sql.parquet.compression.codec", "zstd") \
    .config("spark.driver.memory", "512g") \
    .config("spark.driver.maxResultSize", "256g") \
    .config("spark.memory.offHeap.enabled", "true") \
    .config("spark.memory.offHeap.size", "256g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()

# Log Spark UI URLs for monitoring
ui_web_url = spark.sparkContext.uiWebUrl  # This is the full URL (e.g., http://localhost:4040)
logging.info(f"Spark UI available at: {ui_web_url}")

# Load parquet directory
df = spark.read.parquet(str(tmpdir))
logging.info("Parquet files loaded.")

# Deduplicate
df = df.dropDuplicates(df.columns)  # Replace with actual key columns if available

# Remove old output
shutil.rmtree(partitioned_dir, ignore_errors=True)

# Write output with more partitions to reduce memory pressure
df.write \
    .mode("overwrite") \
    .option("compression", "zstd") \
    .option("maxRecordsPerFile", 10_000_000) \
    .parquet(str(partitioned_dir))
logging.info("Deduplication complete.")
logging.info(f"Partitioned dataset written to {partitioned_dir}")

# Efficient AUC by nprops with memory optimizations
logging.info("Starting AUC calculation...")

# Select only required columns and cache to avoid recomputation
scored = spark.read.parquet(str(partitioned_dir))
scored = scored.select("nprops", "property_token","true_value", "prob_of_1")
scored.cache()

# Grouped processing using Pandas UDF for efficient AUC computation
from pyspark.sql.functions import pandas_udf
from pyspark.sql.types import DoubleType
from typing import Any
import pandas as pd

@pandas_udf(DoubleType())
def compute_auc(value_series: pd.Series, probs_series: pd.Series) -> float:
    try:
        if value_series.nunique() < 2:
            return float("nan")
        return sklearn.metrics.roc_auc_score(value_series, probs_series)
    except Exception:
        return float("nan")


# Process in smaller chunks
auc_df = scored.groupBy(["nprops","property_token"]).agg(compute_auc(F.col("true_value"), F.col("prob_of_1")).alias("auc"))
auc_results = auc_df.orderBy("nprops").toPandas()

# pivot with property_token on the rows and nprops on the columns
auc_results = auc_results.pivot(index="property_token", columns="nprops", values="auc").reset_index()
auc_results
logging.info(f"AUC by nprops:\n{auc_results}")

# Clean up
scored.unpersist()
spark.stop()
