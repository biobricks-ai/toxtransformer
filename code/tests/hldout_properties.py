# purpose: find the unique property tokens in the hldout dataset and count the number of properties per source
# we want to be sure the holdout set is representative of our sources

import pandas as pd, sqlite3, torch, pathlib
import cvae.tokenizer
import cvae.models.multitask_transformer as mt
import logging
import pathlib
from tqdm import tqdm

logdir = pathlib.Path("cache/tests/hldout_properties")
logdir.mkdir(exist_ok=True, parents=True)
logging.basicConfig(level=logging.INFO, filename=logdir / 'hldout_properties.log', filemode='w')

# Setup paths and tokenizer
tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer/')
tokenizer.assay_indexes_tensor = torch.tensor(list(tokenizer.assay_indexes().values()))

# Load dataset (no DDP, just single-process local)
dataset = mt.RotatingModuloSequenceShiftDataset(
    path="cache/build_tensordataset/multitask_tensors/hld",
    tokenizer=tokenizer,
    nprops=5
)

# Extract all property tokens from dataset
property_tokens = []

for i in tqdm(range(len(dataset))):
    _, _, raw_out = dataset[i]
    tokens = raw_out[torch.isin(raw_out, tokenizer.assay_indexes_tensor)].tolist()
    property_tokens.extend(tokens)

# Count unique property tokens
unique_tokens = pd.Series(property_tokens).drop_duplicates().astype(int)

logging.info(f"Found {len(unique_tokens)} unique property tokens.")

# Load prop_src from sqlite
conn = sqlite3.connect('brick/cvae.sqlite')
prop_src = pd.read_sql("""
    SELECT property_token, title, source 
    FROM property p 
    INNER JOIN source s ON p.source_id = s.source_id
""", conn)

# Merge with property tokens
df_tokens = pd.DataFrame({'property_token': unique_tokens})
merged = df_tokens.merge(prop_src, on='property_token', how='left')

# Group by source and count
source_counts = merged['source'].value_counts()
logging.info("\nProperty token counts by source:")
logging.info(source_counts)
