# PYTHONPATH=./ spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///tmp/spark-events code/5_1_eval_multi_properties.py 2> cache/eval_multi_properties/logs/err.log

import pickle
import itertools, uuid, pathlib
import pandas as pd, tqdm, sklearn.metrics, torch, numpy as np, os
import cvae.tokenizer, cvae.models.multitask_transformer as mt, cvae.utils, cvae.models.mixture_experts as me
import logging
from cvae.tokenizer import SelfiesPropertyValTokenizer
from pyspark.sql.functions import col, when, countDistinct
from pyspark.sql import functions as F
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, when
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, log_loss

# Setup logging
logdir = pathlib.Path("cache/eval_multi_properties/logs")   
logdir.mkdir(exist_ok=True, parents=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=logdir / 'eval_multi_properties.log')

# Output directories
outdir = pathlib.Path("cache/eval_multi_properties")
outdir.mkdir(exist_ok=True, parents=True)

tqdm.tqdm.pandas()

# Load tokenizer and Spark session
spark = cvae.utils.get_spark_session()

# Read predictions
outdf = spark.read.parquet("cache/direct_eval_predictions/predictions_parquet")

def calculate_metrics(y_true, y_pred):
    if len(set(y_true)) < 2:
        return -1.0, -1.0, -1.0, -1.0
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

meanpred = outdf.groupBy("nprops", "property_token", "inchi", "value").agg(F.mean("prob_of_1").alias("probs"))
# meanpred.write.parquet((outdir / "multitask_meanpred.parquet").as_posix(), mode="overwrite")

# get dict of inchi to list of inchi to list of (property_token, value_token) from 5_0_2_0_direct_evaluation.py
with open("cache/direct_eval/activity_by_inchi.pkl", "rb") as f:
    activity_by_inchi = pickle.load(f)
inchi_to_nprops = [{"inchi": inchi, "num_known_properties": len(props)} for inchi, props in activity_by_inchi.items()]
inchi_props_df = pd.DataFrame(inchi_to_nprops)
inchi_props_sdf = spark.createDataFrame(inchi_props_df)

large_properties_df = meanpred.groupBy('nprops', 'property_token').agg(
    F.collect_list('value').alias('y_true'),
    F.collect_list('probs').alias('y_pred'),
    countDistinct('inchi').alias('nchem'),
    F.sum(when(col('value') == 1, 1).otherwise(0)).alias('NUM_POS'),
    F.sum(when(col('value') == 0, 1).otherwise(0)).alias('NUM_NEG'))

metrics_df = large_properties_df.repartition(800) \
    .withColumn('metrics', calculate_metrics_udf(F.col('y_true'), F.col('y_pred'))) \
    .select('nprops', 'property_token', 
        col('metrics.AUC').alias('AUC'), 
        col('metrics.ACC').alias('ACC'), 
        col('metrics.BAC').alias('BAC'), 
        col('metrics.cross_entropy_loss').alias('cross_entropy_loss'), 'NUM_POS', 'NUM_NEG', 'nchem')

metrics_df.write.parquet((outdir / "multitask_metrics.parquet").as_posix(), mode="overwrite")
