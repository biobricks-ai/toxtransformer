# purpose: preprocess chemharmony data to create SELFIES, a tokenizer, and a Parquet file for conversion to a TensorDataset.
# dependencies: chemharmony
# outputs: 
#     - cache/preprocess/substances.parquet - Parquet file containing processed substance data with SELFIES, encoded SELFIES, InChI, and SMILES.
# cmd: spark-submit \
#   --master local[240] \
#   --driver-memory 512g \
#   --conf spark.eventLog.enabled=true \
#   --conf spark.eventLog.dir=file:///tmp/spark-events \
#   code/1_1_preprocess.py

import biobricks
import pyspark.sql, pyspark.sql.functions as F, pyspark.ml.feature
import cvae.tokenizer.selfies_tokenizer, cvae.utils, cvae.spark_helpers as H
import pathlib
import logging
import os
from pyspark.sql import SparkSession
import os
import multiprocessing
import psutil

# Set up logging
logdir = pathlib.Path('cache/preprocess/log')
logdir.mkdir(parents=True, exist_ok=True)
cvae.utils.setup_logging(logdir / 'preprocess.log', logging)

# Ensure Spark event logging directory exists if needed by your Spark config
os.makedirs("/tmp/spark-events", exist_ok=True)

outdir: pathlib.Path = pathlib.Path('cache/preprocess')
outdir.mkdir(parents=True, exist_ok=True)

# Configure session parameters
total_cores = os.cpu_count()
total_ram_gb = psutil.virtual_memory().total / (1024 ** 3)
cores_max = min(192, total_cores)                       # leave headroom
executor_cores = 4
executor_mem = f"{min(256, total_ram_gb // 6)}g"        # ~256 GB per executor
driver_mem = f"{min(512, total_ram_gb // 3)}g"          # ~512 GB for driver
shuffle_partitions = cores_max * 2                      # good parallelism

spark = SparkSession.builder \
    .appName("ChemharmonyPreprocessing") \
    .config("spark.driver.memory", driver_mem) \
    .config("spark.executor.memory", executor_mem) \
    .config("spark.executor.cores", str(executor_cores)) \
    .config("spark.cores.max", str(cores_max)) \
    .config("spark.task.cpus", "1") \
    .config("spark.driver.maxResultSize", "8g") \
    .config("spark.sql.shuffle.partitions", str(shuffle_partitions)) \
    .config("spark.sql.files.maxPartitionBytes", str(128 * 1024 * 1024)) \
    .config("spark.python.profile", "true") \
    .config("spark.eventLog.enabled", "true") \
    .config("spark.eventLog.dir", "file:///tmp/spark-events") \
    .config("spark.network.timeout", "600s") \
    .config("spark.executor.heartbeatInterval", "60s") \
    .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# GET CHEMHARMONY SUBSTANCES ===============================================================
chemharmony = biobricks.assets('chemharmony')
rawsubstances = spark.read.parquet(chemharmony.substances_parquet).select("sid", "source", "data")
rawsubstances.repartition(128).write.parquet((outdir / 'substances.parquet').as_posix(), mode='overwrite')
logging.info(f"wrote {outdir / 'substances.parquet'}")

# GET SELFIES ===============================================================================

## Extract INCHI and SMILES from the data json column. It has a few different names for the same thing
substances = spark.read.parquet((outdir / 'substances.parquet').as_posix()) \
    .withColumn("rawinchi", F.get_json_object("data", "$.inchi")) \
    .withColumn("ligand_inchi", F.get_json_object("data", "$.Ligand InChI")) \
    .withColumn("rawsmiles", F.get_json_object("data", "$.SMILES")) \
    .withColumn("ligand_smiles", F.get_json_object("data", "$.Ligand SMILES")) \
    .withColumn("inchi", F.coalesce("rawinchi", "ligand_inchi")) \
    .withColumn("smiles", F.coalesce("rawsmiles", "ligand_smiles")) \
    .select("sid", "source", "inchi", "smiles")

substances = substances.withColumn("smiles", F.coalesce("smiles", H.inchi_to_smiles_udf("inchi")))
substances = substances.filter(substances.smiles.isNotNull())
substances = substances.withColumn("selfies", H.smiles_to_selfies_udf("smiles"))
substances = substances.select("sid", "inchi", "smiles", "selfies").distinct()
substances = substances.filter(substances.selfies.isNotNull())

# save an intermediate parquet file
substances.write.parquet((outdir / 'substances2.parquet').as_posix(), mode='overwrite')
logging.info(f"wrote {outdir / 'substances2.parquet'}")

spark.stop()
