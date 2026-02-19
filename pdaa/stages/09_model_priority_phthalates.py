import json
import time
import asyncio
import pandas as pd
import pathlib
import rdkit, rdkit.Chem
from itertools import product, islice
from tqdm.asyncio import tqdm as tqdm_async
from tqdm import tqdm
from tenacity import retry, stop_after_attempt, wait_exponential
import biobricks as bb
import sqlite3
import sys
import itertools as it
import random
import threading
from tqdm.asyncio import tqdm_asyncio
from typing import Iterator, Tuple
sys.path.append('./')
import stages.utils.chemprop as chemprop
import stages.utils.pdaa as pdaa

tqdm.pandas()

# Setup paths and caches
brickdir = pathlib.Path('brick')

cptransformer = bb.assets('chemprop-transformer').cvae_sqlite
cpsqlite = sqlite3.connect(cptransformer)
property_tokens = pd.read_sql_query("SELECT property_token FROM property", cpsqlite)['property_token'].unique().tolist()
property_tokens = sorted(property_tokens)

raw_df = pd.read_parquet('cache/priority_phthalates/priority_phthalates.parquet')
top_df = raw_df.sort_values(by='max_similarity', ascending=False)[['inchi', 'max_similarity']].drop_duplicates()
inchi_list = top_df['inchi'].unique().tolist()

# BATCH RUN ==============================================================
def get_missing(inchi_list):
    inchi_tok_pairs = [(inchi, tok) for inchi in inchi_list for tok in property_tokens]
    missing_inchi = set()
    with sqlite3.connect(brickdir / 'predictions.sqlite') as conn:
        for inchi, property_token in inchi_tok_pairs:
            if inchi in missing_inchi:
                continue
            cursor = conn.execute('SELECT * FROM predictions WHERE inchi = ? AND property_token = ?', (inchi, property_token))
            exists = cursor.fetchone() is not None
            if not exists:
                missing_inchi.add(inchi)
    return missing_inchi

# BUILD PREDICTION FUNCTION =====================================================

async def process_batches():
    BATCH_SIZE = 16*100
    batch_generator = it.batched(iter(inchi_list), BATCH_SIZE)
    num_batches = len(inchi_list) // BATCH_SIZE
    
    num_errors = 0
    sqlite_lock = threading.Lock()
    semaphore = asyncio.Semaphore(16)
    for batch in tqdm(batch_generator, total=num_batches, desc="Processing batches"):
        inchi_batch : Iterator[str] = get_missing(batch)
        print(f'{len(inchi_batch)} missing to predict')

        predictions = [pdaa.async_predict_all(inchi, semaphore) for inchi in inchi_batch]
        rawresults = await tqdm_asyncio.gather(*predictions, return_exceptions=True, desc="predicting...")

        errors = [res for res in rawresults if isinstance(res, Exception)]
        num_errors += len(errors)

        flatres = [item for sublist in rawresults for item in sublist if not isinstance(item, Exception)]
        results = [(res['inchi'], res['property_token'], res['value']) for res in flatres]
        pdaa.add_predictions(results, sqlite_lock)
        print(f"added {len(results)} predictions")

asyncio.run(process_batches())