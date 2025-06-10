# PYTHONPATH=./ python code/5_0_2_direct_evaluation.py


import time
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from rdkit import Chem
import selfies
import sqlite3
import random
import os
import pathlib
import pickle
import logging
from pathlib import Path
import gc
from cvae.tokenizer import SelfiesPropertyValTokenizer
from cvae.spark_helpers import smiles_to_selfies_safe, is_valid_smiles
import cvae.models.mixture_experts as me
import cvae.models.multitask_transformer as mt
from cvae.models.datasets.inmemory_sequence_shift_dataset import PreloadedSequenceShiftDataset, PreloadedSequenceShiftWrapper
from cvae.simplecache import simplecache

# --- Cache and log directory setup ---
cachedir = Path("cache/direct_eval")
cachedir.mkdir(parents=True, exist_ok=True)
(cachedir / "hldout_eval_tensors").mkdir(parents=True, exist_ok=True)

logdir = cachedir / "log"
logdir.mkdir(parents=True, exist_ok=True)

# --- Shared setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = me.MoE.load("cache/train_multitask_transformer_parallel/models/moe").to(DEVICE).eval()
TOKENIZER = MODEL.tokenizer
VALUE_TOKEN_MAP = TOKENIZER.value_indexes()
VALUE_TO_LABEL = {v: k for k, v in VALUE_TOKEN_MAP.items()}
VALUE_TOKENS = [VALUE_TOKEN_MAP[0], VALUE_TOKEN_MAP[1]]

# --- Load activity_by_inchi before spawning Pool (safe with fork) ---
activity_cache_path = cachedir / "activity_by_inchi.pkl"
if activity_cache_path.exists():
    with open(activity_cache_path, "rb") as f:
        activity_by_inchi = pickle.load(f)
else:
    CONN = sqlite3.connect("brick/cvae.sqlite")
    activity_df = pd.read_sql("SELECT inchi, property_token, value_token FROM activity", CONN)
    activity_by_inchi = defaultdict(list)
    rows = list(activity_df.itertuples(index=False))
    for row in tqdm(rows, desc="Building activity_by_inchi"):
        activity_by_inchi[row.inchi].append((row.property_token, row.value_token))
    with open(activity_cache_path, "wb") as f:
        pickle.dump(activity_by_inchi, f)

# --- Helpers ---
def inchi_pv_to_ctx_out(inchi, ptok, vtok, nprops=16, nsamples=1):
    pv_pairs = [(p, v) for p, v in activity_by_inchi[inchi] if not (p == ptok and v == vtok)]

    max_len = nprops * 2 + 2

    def sample_tokens():
        k = min(nprops - 1, len(pv_pairs))
        if k == 0:
            sampled = []
        else:
            sampled = sum(random.sample(pv_pairs, k), ())
        tokens = [TOKENIZER.SEP_IDX] + list(map(int, sampled)) + [ptok, vtok, TOKENIZER.END_IDX]
        tokens += [TOKENIZER.PAD_IDX] * (max_len - len(tokens))
        tokens = torch.tensor(tokens)
        return torch.cat([torch.tensor([1]), tokens[:-1]]), tokens

    samples = [sample_tokens() for _ in range(nsamples)]
    ctx, out = zip(*samples)
    ctx, out = torch.stack(ctx), torch.stack(out)

    return ctx, out

# @simplecache(cachedir / "selfies_tokens_to_inchi")
def selfies_tokens_to_inchi(selfies_tokens):
    try:
        smiles = selfies.decoder(selfies_tokens)
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchi(mol)
    except Exception:
        return None

PRELOADED_DATASET = PreloadedSequenceShiftDataset("cache/build_tensordataset/multitask_tensors/hld", tokenizer=TOKENIZER)
def process_property_nprops_shard(top_outdir, nprops, target_prop):
    outdir = top_outdir / f"nprops_{nprops}"
    outdir.mkdir(parents=True, exist_ok=True)
    outpath = str(outdir / f"property_{target_prop}_nprops_{nprops}.pt")
    if os.path.exists(outpath):
        return f"Skipping existing file: {outpath}"
    

    invalid_inchi_count = 0
    ds = PreloadedSequenceShiftWrapper(PRELOADED_DATASET, nprops=1, assay_filter=[target_prop])

    try:
        batch_inp, batch_ctx, batch_out, metadata = [], [], [], []

        for i in range(len(ds)):
            r1i, _, r1o = ds[i]
            selfies_tokens = TOKENIZER.selfies_tokenizer.indexes_to_selfies(r1i.tolist())
            inchi = selfies_tokens_to_inchi(selfies_tokens)
            if not inchi or inchi not in activity_by_inchi:
                invalid_inchi_count += 1
                print(f"Error in inchi_pv_to_ctx_out: {inchi} {invalid_inchi_count} out of {i}")
                continue

            ptok, vtok = r1o[1].item(), r1o[2].item()
            try:
                # Scale number of samples with nprops to increase context diversity, capped at 32
                nsamples = min(max(1, nprops // 4), 32)
                ctx, out = inchi_pv_to_ctx_out(inchi, ptok, vtok, nprops=nprops, nsamples=nsamples)
            except Exception as e:
                print(f"Error in inchi_pv_to_ctx_out: {e}")
                continue

            batch_inp.append(r1i)
            batch_ctx.append(ctx)
            batch_out.append(out)
            metadata.append((inchi, nprops, ptok, vtok))

        if batch_inp:
            torch.save({
                "inp": torch.stack(batch_inp),
                "ctx": torch.stack(batch_ctx),
                "out": torch.stack(batch_out),
                "meta": metadata
            }, outpath)

        return f"✅ Finished property {target_prop} nprops {nprops} with {len(batch_inp)} examples"

    except Exception as e:
        return f"❌ Failed property {target_prop} nprops {nprops}: {e}"

import psutil
def shard_worker(args):
    result = process_property_nprops_shard(*args)
    rss = psutil.Process(os.getpid()).memory_info().rss / 1e9
    print(f"[PID {os.getpid()}] Memory usage: {rss:.2f} GB")
    return result

def run_worker(args):
    print(shard_worker(args))

# --- Main ---
from multiprocessing import Process
if __name__ == "__main__":
    
    MAX_CONCURRENT = 100

    bp = pd.read_parquet("cache/get_benchmark_properties/benchmark_properties.parquet")
    target_props = list(set(bp['property_token'].tolist()))
    outdir = cachedir / "hldout_eval_tensors"
    outdir.mkdir(parents=True, exist_ok=True)
    nprops_list = [1, 5, 10, 20, 50, 100]

    worker_args = [
        (outdir, nprops, target_prop)
        for nprops in nprops_list
        for target_prop in target_props
    ]

    active_processes = []

    for arg in tqdm(worker_args, desc="Spawning property+nprops shard workers"):
        # Wait for an available slot
        while len(active_processes) >= MAX_CONCURRENT:
            for p in active_processes:
                if not p.is_alive():
                    p.join()
            active_processes = [p for p in active_processes if p.is_alive()]
            time.sleep(0.5)

        # Launch next process
        p = Process(target=run_worker, args=(arg,))
        p.start()
        active_processes.append(p)

    # Final cleanup
    for p in active_processes:
        p.join()