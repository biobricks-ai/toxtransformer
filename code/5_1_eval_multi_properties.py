"""
rm -rf cache/eval_multi_properties/
for SPLIT in {0..5}; do
  INDIR="cache/consolidate_evaluations/${SPLIT}/multitask_predictions.parquet"
  OUTDIR="cache/eval_multi_properties/${SPLIT}"
  mkdir -p "$OUTDIR/logs"
  
  SPLIT=$SPLIT INDIR="$INDIR" OUTDIR="$OUTDIR" PYTHONPATH=./ \
  spark-submit --master local[240] \
    --driver-memory 512g \
    --conf spark.driver.maxResultSize=256g \
    --conf spark.eventLog.enabled=true \
    --conf spark.eventLog.dir=file:///tmp/spark-events \
    code/5_1_eval_multi_properties.py \
    2> "$OUTDIR/logs/err.log"
done
"""

import os
import pathlib
import pandas as pd, tqdm, sklearn.metrics, torch, numpy as np
import cvae.tokenizer, cvae.models.multitask_transformer as mt, cvae.utils, cvae.models.mixture_experts as me
import logging
import shutil

from cvae.tokenizer import SelfiesPropertyValTokenizer
from pyspark.sql.functions import col, when, countDistinct, lit
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, when
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, log_loss

# Get environment variables
SPLIT = os.getenv("SPLIT")
INDIR = os.getenv("INDIR")
OUTDIR = os.getenv("OUTDIR")

# Output directories
outdir = pathlib.Path(OUTDIR)
outdir.mkdir(exist_ok=True, parents=True)

# Setup logging
logdir = outdir / "logs"
logdir.mkdir(exist_ok=True, parents=True)
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s', 
    filename=logdir / 'eval_multi_properties.log',
    filemode='w'
)

logging.info(f"Starting eval_multi_properties for SPLIT={SPLIT}")

tqdm.tqdm.pandas()

# Load Spark session
pathlib.Path("/tmp/spark-events").mkdir(exist_ok=True, parents=True)
spark = SparkSession.builder \
    .appName(f"eval_multi_properties_Split{SPLIT}") \
    .master("local[240]") \
    .config("spark.driver.memory", "512g") \
    .config("spark.driver.maxResultSize", "256g") \
    .config("spark.sql.shuffle.partitions", "128") \
    .config("spark.sql.files.maxPartitionBytes", str(64 * 1024 * 1024)) \
    .config("spark.network.timeout", "600s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

# Log Spark UI
ui_web_url = spark.sparkContext.uiWebUrl
logging.info(f"Spark UI available at: {ui_web_url}")

# Read predictions
logging.info(f"Reading predictions from {INDIR}")
outdf = spark.read.parquet(INDIR)
logging.info(f"Loaded {outdf.count()} records for SPLIT={SPLIT}")

# ---- Log-odds pooling setup ----
EPS = 1e-6  # clip to avoid infinities from exact 0/1
TEMPERATURE = float(os.getenv("LOGIT_POOL_TEMPERATURE", "1.0"))  # optional damping
logging.info(f"Using log-odds pooling with TEMPERATURE={TEMPERATURE}")

# clip p into (EPS, 1-EPS)
p = col("prob_of_1")
p_clipped = F.when(p <= lit(EPS), lit(EPS)) \
             .when(p >= lit(1.0 - EPS), lit(1.0 - EPS)) \
             .otherwise(p)

# logit(p) = ln(p/(1-p))
logit = F.log(p_clipped / (1.0 - p_clipped))

# Aggregate by summing logits within each group
group_cols = ["nprops", "minprops", "numprops", "property_token", "chemical_id", "true_value"]

sum_logits_df = outdf.groupBy(*group_cols).agg(
    F.sum(logit).alias("sum_logit"),
    F.count(F.lit(1)).alias("num_votes")
)

# sigmoid(sum_logit / T)
meanpred = sum_logits_df.withColumn(
    "probs",
    1.0 / (1.0 + F.exp(-(col("sum_logit") / lit(TEMPERATURE))))
).cache()

logging.info(f"Computed pooled predictions: {meanpred.count()} records")

# ---- Metrics ----
def calculate_metrics(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # Must be 0/1 labels
    y_true_binary = (y_true > 0).astype(int)

    # Use thresholded predictions for accuracy/BAC
    y_pred_binary = (y_pred > 0.5).astype(int)

    auc = float(roc_auc_score(y_true_binary, y_pred))
    acc = float(accuracy_score(y_true_binary, y_pred_binary))
    bac = float(balanced_accuracy_score(y_true_binary, y_pred_binary))
    ce_loss = float(log_loss(y_true_binary, y_pred))
    return auc, acc, bac, ce_loss

calculate_metrics_udf = F.udf(calculate_metrics, "struct<AUC:double, ACC:double, BAC:double, cross_entropy_loss:double>")

from pyspark.sql.window import Window

# Aggregate across chemicals per (nprops, property_token, minprops)
large_properties_df = meanpred \
    .groupBy('nprops', 'property_token', 'minprops') \
    .agg(
        F.collect_list('true_value').alias('y_true'),
        F.collect_list('probs').alias('y_pred'),
        countDistinct('chemical_id').alias('nchem'),
        F.sum(when(col('true_value') == 1, 1).otherwise(0)).alias('NUM_POS'),
        F.sum(when(col('true_value') == 0, 1).otherwise(0)).alias('NUM_NEG')
    ) \
    .filter((col('NUM_POS') >= 10) & (col("NUM_NEG") >= 10))

logging.info(f"Filtered to {large_properties_df.count()} large properties")

large_properties_path = outdir / "multitask_large_properties.parquet"
large_properties_df.write.parquet(str(large_properties_path), mode="overwrite")
logging.info(f"Wrote large properties to {large_properties_path}")

metrics_df = large_properties_df.repartition(800) \
    .withColumn('metrics', calculate_metrics_udf(F.col('y_true'), F.col('y_pred'))) \
    .select('nprops', 'minprops', 'property_token',
        col('metrics.AUC').alias('AUC'),
        col('metrics.ACC').alias('ACC'),
        col('metrics.BAC').alias('BAC'),
        col('metrics.cross_entropy_loss').alias('cross_entropy_loss'),
        'NUM_POS', 'NUM_NEG', 'nchem')

logging.info("Computing metrics summary by nprops and minprops...")
summary_df = metrics_df.orderBy('nprops') \
    .groupBy('nprops') \
    .pivot('minprops') \
    .agg(F.mean("AUC")) \
    .orderBy('nprops')

summary_df.show()
logging.info("Summary table shown above")

metrics_path = outdir / "multitask_metrics.parquet"
metrics_df.write.parquet(str(metrics_path), mode="overwrite")
logging.info(f"Wrote metrics to {metrics_path}")

# Stop Spark
spark.stop()
logging.info(f"Evaluation complete for SPLIT={SPLIT}")