import pyspark.sql, pyspark.sql.functions as F, pyspark.ml.feature
import cvae.tokenizer.selfies_tokenizer, cvae.utils, cvae.spark_helpers as H
import pathlib
import logging

# cmd: spark-submit \
#   --master local[240] \
#   --driver-memory 512g \
#   --conf spark.eventLog.enabled=true \
#   --conf spark.eventLog.dir=file:///tmp/spark-events \
#   code/1_2_preprocess_tokenizer.py


# Set up logging
logdir = pathlib.Path('cache/preprocess_tokenizer/log')
logdir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', filename=logdir / 'preprocess_tokenizer.log', filemode='w')

outdir : pathlib.Path = pathlib.Path('cache/preprocess_tokenizer')
outdir.mkdir(parents=True, exist_ok=True)

# Initialize Spark session at the start before any Spark operations
spark = cvae.utils.get_spark_session()
spark.sparkContext.setLogLevel("ERROR")

# BUILD TOKENIZER ===========================================================================
substances2 = spark.read.parquet('cache/preprocess/substances2.parquet')
tokenizer = cvae.tokenizer.selfies_tokenizer.SelfiesTokenizer().fit(substances2, 'selfies')
substances3 = tokenizer.transform(substances2, 'selfies', 'encoded_selfies')

tokenizer.save((outdir / 'selfies_tokenizer.json').as_posix())
substances3.write.parquet(str(outdir / 'substances.parquet'), mode='overwrite')
logging.info(f"wrote {outdir / 'substances.parquet'}")
