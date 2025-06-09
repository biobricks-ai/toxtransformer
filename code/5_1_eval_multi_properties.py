# PYTHONPATH=./ spark-submit --master local[240] --driver-memory 512g --conf spark.eventLog.enabled=true --conf spark.eventLog.dir=file:///tmp/spark-events code/5_1_eval_multi_properties.py 2> cache/eval_multi_properties/logs/err.log

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
model = me.MoE.load("cache/train_multitask_transformer_parallel/models/moe")
tokenizer : SelfiesPropertyValTokenizer = model.tokenizer
# tokenizer : SelfiesPropertyValTokenizer = me.MoE.load("brick/moe").tokenizer
spark = cvae.utils.get_spark_session()

# Read predictions
outdf = spark.read.parquet("cache/consolidate_evaluations/multitask_predictions.parquet")
outdf = outdf.filter(F.col('assay') == 3851)
test = outdf.toPandas()
test['probval'] = test['probs'].apply(lambda x: x > .5)

test[['value','probval']].value_counts()

test2 = test[test['chemical_id'] == test['chemical_id'].iloc[0]][['chemical_id','probs','value']]

test2[['value','probval']].value_counts()
# Calculate metrics
value_indexes = list(tokenizer.value_indexes().values())
val0_index, val1_index = value_indexes[0], value_indexes[1]

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

meanpred = outdf.groupBy("nprops", "assay", "chemical_id", "value")\
    .agg(F.mean("probs").alias("probs"))\
    .withColumn("value", F.col("value").cast("int"))\
    .withColumn("value", F.when(F.col("value") == val0_index, 0).otherwise(1))

meanpred.write.parquet((outdir / "multitask_meanpred.parquet").as_posix(), mode="overwrite")

large_properties_df = meanpred.groupBy('nprops', 'assay').agg(
    F.collect_list('value').alias('y_true'),
    F.collect_list('probs').alias('y_pred'),
    countDistinct('chemical_id').alias('nchem'),
    F.sum(when(col('value') == 1, 1).otherwise(0)).alias('NUM_POS'),
    F.sum(when(col('value') == 0, 1).otherwise(0)).alias('NUM_NEG'))

metrics_df = large_properties_df.repartition(800) \
    .withColumn('metrics', calculate_metrics_udf(F.col('y_true'), F.col('y_pred'))) \
    .select('nprops', 'assay', 
        col('metrics.AUC').alias('AUC'), 
        col('metrics.ACC').alias('ACC'), 
        col('metrics.BAC').alias('BAC'), 
        col('metrics.cross_entropy_loss').alias('cross_entropy_loss'), 
        'NUM_POS', 'NUM_NEG',
        'nchem')

metrics_df.write.parquet((outdir / "multitask_metrics.parquet").as_posix(), mode="overwrite")
