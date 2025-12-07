# purpose: preprocess chemharmony data to create SELFIES, a tokenizer, and a Parquet file for conversion to a TensorDataset.
# dependencies: chemharmony
# outputs: 
#     - cache/preprocess/substances.parquet - Parquet file containing processed substance data with SELFIES, encoded SELFIES, InChI, and SMILES.
import biobricks
import pyspark.sql, pyspark.sql.functions as F, pyspark.ml.feature
import cvae.tokenizer.selfies_tokenizer, cvae.utils, cvae.spark_helpers as H
import pathlib
import logging

# Set up logging
logdir = pathlib.Path('log')
logdir.mkdir(parents=True, exist_ok=True)
cvae.utils.setup_logging(logdir / 'preprocess.log', logging)

outdir : pathlib.Path = pathlib.Path('cache/preprocess')
outdir.mkdir(parents=True, exist_ok=True)

spark = cvae.utils.get_spark_session()

# GET CHEMHARMONY SUBSTANCES ===============================================================
chemharmony = biobricks.assets('chemharmony')
rawsubstances = spark.read.parquet(chemharmony.substances_parquet).select("sid","source","data")
rawsubstances.repartition(16).write.parquet((outdir / 'substances.parquet').as_posix(), mode='overwrite')
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
    .select("sid","source","inchi","smiles")

substances = substances.withColumn("smiles", F.coalesce("smiles", H.inchi_to_smiles_udf("inchi")))
substances = substances.filter(substances.smiles.isNotNull())
substances = substances.withColumn("selfies", H.smiles_to_selfies_udf("smiles"))
substances = substances.select('sid', 'inchi', 'smiles', 'selfies').distinct()
substances = substances.filter(substances.selfies.isNotNull())

# save an intermediate parquet file
substances.write.parquet((outdir / 'substances2.parquet').as_posix(), mode='overwrite')
logging.info(f"wrote {outdir / 'substances2.parquet'}")

spark.stop()