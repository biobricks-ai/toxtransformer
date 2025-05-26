import glob
import logging
import pathlib
import shutil
import pandas as pd
import pyarrow.dataset as ds
from tqdm import tqdm
import sklearn.metrics

# Set these paths explicitly or via argparse/env
outdir = pathlib.Path("cache/consolidate_evaluations")
tmpdir = pathlib.Path("cache/generate_evaluations") / "temp"

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", filename=outdir / "consolidate_evaluations.log", filemode="w")
logging.info("Starting finalize_output.")

# Load and concatenate all temp parquet files
parquet_files = glob.glob(str(tmpdir / "*.parquet"))
logging.info(f"Found {len(parquet_files)} parquet files.")
all_dfs = [pd.read_parquet(file) for file in tqdm(parquet_files, desc="Reading Parquets")]
df = pd.concat(all_dfs, ignore_index=True)
logging.info(f"Concatenated DataFrame shape: {df.shape}")

# deduplicate
logging.info(f"Deduplicating DataFrame...")
df = df.drop_duplicates()
logging.info(f"Deduplicated DataFrame shape: {df.shape}")

# Save partitioned dataset directly
logging.info(f"Saving partitioned dataset...")
partitioned_dir = outdir / "multitask_predictions.parquet"
shutil.rmtree(partitioned_dir, ignore_errors=True)
partitioned_dir.mkdir(parents=True, exist_ok=True)

ds.write_dataset(
    data=df,
    base_dir=partitioned_dir,
    format="parquet",
    file_options=ds.ParquetFileFormat().make_write_options(compression="zstd", compression_level=9),
    max_rows_per_file=25_000_000,
    existing_data_behavior="overwrite_or_ignore",
    basename_template="part-{i}.parquet",
)
logging.info(f"Saved partitioned dataset to: {partitioned_dir}")

# Calculate AUC score
logging.info("Calculating AUC score...")
final_df = ds.dataset(partitioned_dir, format="parquet").to_table().to_pandas()
auc_by_nprop = final_df.groupby('nprops').apply(
    lambda x: sklearn.metrics.roc_auc_score(x['value'], x['probs'])
)
logging.info(f"AUC by number of properties:\n{auc_by_nprop}")
