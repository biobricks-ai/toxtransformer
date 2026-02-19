import sys
import threading
import random
sys.path.append('./')
import stages.utils.chemprop as chemprop
import stages.utils.pdaa as pdaa
import asyncio
import pandas as pd
import pathlib
import itertools as it
import biobricks as bb
import sqlite3
import random

from tqdm.asyncio import tqdm as tqdm_async, tqdm_asyncio
from tqdm import tqdm

# SETUP PATHS AND CACHES ========================================================
brickdir = pathlib.Path('brick')
cachedir = pathlib.Path('cache/model_phthalates')
cachedir.mkdir(parents=True, exist_ok=True)

# LOAD CHEMPROP-TRANSFORMER PROPERTY TOKENS =====================================
with sqlite3.connect(bb.assets('chemprop-transformer').cvae_sqlite) as con:
    property_tokens = pd.read_sql_query("SELECT property_token FROM property", con)['property_token'].tolist()
    property_tokens = sorted(list(set(property_tokens)))

# LOAD ZINC PHTALATES DATA =====================================================
raw_df = pd.read_parquet('cache/zinc_phthalates/zinc_phthalates.parquet')
inchi_list = raw_df['inchi'].unique().tolist()

# BATCH RUN ==============================================================
def get_missing(inchi_tok_pairs):
    with sqlite3.connect(brickdir / 'predictions.sqlite') as conn:
        results = []
        for inchi, property_token in inchi_tok_pairs:
            cursor = conn.execute('SELECT * FROM predictions WHERE inchi = ? AND property_token = ?', (inchi, property_token))
            exists = cursor.fetchone() is not None
            if not exists:
                results.append((inchi, property_token))
    return results

# BUILD PREDICTION FUNCTION =====================================================

async def process_batches():
    BATCH_SIZE = 100000
    num_combinations = len(inchi_list) * len(property_tokens)
    rand_inchi, rand_tok = random.sample(inchi_list, len(inchi_list)), random.sample(property_tokens, len(property_tokens))
    tuple_generator = it.product(rand_inchi, rand_tok)
    batch_generator = it.batched(tuple_generator, BATCH_SIZE)
    num_batches = num_combinations // BATCH_SIZE
    
    sqlite_lock = threading.Lock()
    semaphore = asyncio.Semaphore(40)

    for batch in tqdm(batch_generator, total=num_batches, desc="Processing batches"):
        new_inchi_tok_pairs = get_missing(batch)
        print(f"new inchi-tok pairs: {len(new_inchi_tok_pairs)}")
        predictions = [pdaa.async_predict(inchi, tok, semaphore) for inchi, tok in new_inchi_tok_pairs]
        results = await tqdm_asyncio.gather(*predictions, desc="predicting...")
        pdaa.add_predictions(results, sqlite_lock)
        print(f"Processed batch: {len(results)}")
    
asyncio.run(process_batches())
