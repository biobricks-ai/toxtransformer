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

from cvae.tokenizer import SelfiesPropertyValTokenizer
from cvae.spark_helpers import smiles_to_selfies_safe, is_valid_smiles
import cvae.models.mixture_experts as me
import cvae.models.multitask_transformer as mt
from cvae.simplecache import simplecache

import selfies
from rdkit import Chem
from multiprocessing import Pool
import os
from pathlib import Path

# Create required logging directory for tracking shard progress
log_dir = Path("cache/direct_eval/log")
log_dir.mkdir(parents=True, exist_ok=True)

# Show the path for confirmation
log_dir.resolve()

# --- Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = me.MoE.load("cache/train_multitask_transformer_parallel/models/moe").to(DEVICE).eval()
TOKENIZER = MODEL.tokenizer
VALUE_TOKEN_MAP = TOKENIZER.value_indexes()
VALUE_TO_LABEL = {v: k for k, v in VALUE_TOKEN_MAP.items()}
VALUE_TOKENS = [VALUE_TOKEN_MAP[0], VALUE_TOKEN_MAP[1]]

# --- Cache setup ---
cache_dir = pathlib.Path("cache/direct_eval")
cache_dir.mkdir(parents=True, exist_ok=True)
activity_cache_path = cache_dir / "activity_by_inchi.pkl"
results_path = cache_dir / "records.parquet"

# --- Load or build activity_by_inchi ---
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
@simplecache(cache_dir / "inchi_pv_to_ctx_out")
def inchi_pv_to_ctx_out(inchi, ptok, vtok, nprops=16, nsamples=1):
    pv_pairs = [(p, v) for p, v in activity_by_inchi[inchi] if not (p == ptok and v == vtok)]
    max_len = nprops * 2 + 2

    def sample_tokens():
        sampled = sum(random.sample(pv_pairs, min(nprops - 1, len(pv_pairs))), ())
        tokens = [TOKENIZER.SEP_IDX] + list(map(int, sampled)) + [ptok, vtok, TOKENIZER.END_IDX]
        tokens += [TOKENIZER.PAD_IDX] * (max_len - len(tokens))
        tokens = torch.tensor(tokens)
        return torch.cat([torch.tensor([1]), tokens[:-1]]), tokens

    samples = [sample_tokens() for _ in range(nsamples)]
    ctx, out = zip(*samples)
    ctx, out = torch.stack(ctx), torch.stack(out)

    return ctx, out

@simplecache(cache_dir / "selfies_tokens_to_inchi")
def selfies_tokens_to_inchi(selfies_tokens):
    try:
        smiles = selfies.decoder(selfies_tokens)
        mol = Chem.MolFromSmiles(smiles)
        return Chem.MolToInchi(mol)
    except Exception as e:
        return None

# --- BUILD pt FILES ---
def process_shard_safe(start_idx, end_idx):
    try:
        shard_name = f"shard_{start_idx}_{end_idx}"
        out_path = cache_dir / f"{shard_name}.pt"
        log_path = cache_dir / "log" / f"{shard_name}.log"

        if out_path.exists() and log_path.exists():
            return f"🟡 Skipped {shard_name}"

        ds = mt.SequenceShiftDataset("cache/build_tensordataset/multitask_tensors/hld", tokenizer=TOKENIZER, nprops=1)
        batch_inp, batch_ctx, batch_out, metadata = [], [], [], []

        for i in range(start_idx, end_idx):
            r1i, _, r1o = ds[i]
            selfies_tokens = TOKENIZER.selfies_tokenizer.indexes_to_selfies(r1i.tolist())
            inchi = selfies_tokens_to_inchi(selfies_tokens)
            if not inchi or inchi not in activity_by_inchi:
                continue

            ptok, vtok = r1o[1].item(), r1o[2].item()
            try:
                ctx, out = inchi_pv_to_ctx_out(inchi, ptok, vtok)
            except Exception:
                continue

            batch_inp.append(r1i)
            batch_ctx.append(ctx)
            batch_out.append(out)
            metadata.append((inchi, ptok, vtok))

        if batch_inp:
            torch.save({
                "inp": torch.stack(batch_inp),
                "ctx": torch.stack(batch_ctx),
                "out": torch.stack(batch_out),
                "meta": metadata
            }, out_path)
            with open(log_path, "w") as f:
                f.write(f"✅ Saved {len(batch_inp)} examples to {out_path}\n")

        return f"✅ Finished {shard_name} with {len(batch_inp)} examples"

    except Exception as e:
        return f"❌ Failed {start_idx}-{end_idx}: {e}"

# --- Parallel Shard Execution with Global Progress ---
from multiprocessing import Pool

def chunkify(n, chunksize):
    return [(i, min(i + chunksize, n)) for i in range(0, n, chunksize)]

if __name__ == "__main__":
    total_len = len(mt.SequenceShiftDataset("cache/build_tensordataset/multitask_tensors/hld", tokenizer=TOKENIZER, nprops=1))
    chunksize = 100_000
    shards = chunkify(total_len, chunksize)

    with Pool(processes=8) as pool:  # Adjust based on core count
        for result in tqdm(pool.imap_unordered(lambda args: process_shard_safe(*args), shards), total=len(shards), desc="Processing shards"):
            print(result)


# # EVALUATE =========================================================================================================

# ds = mt.SequenceShiftDataset(path="cache/build_tensordataset/multitask_tensors/hld",tokenizer=TOKENIZER,nprops=1)
# for i in tqdm(range(len(ds)), desc="Building selfies_tokens_to_inchi"):
#     r1i, _, r1o = ds[i]
#     selfies_tokens = TOKENIZER.selfies_tokenizer.indexes_to_selfies(r1i.tolist())
#     inchi = selfies_tokens_to_inchi(selfies_tokens)

# records = []
# nsamples = 5
# nprops = 100

# for i in tqdm(range(len(ds)), desc="Evaluating dataset"):
#     r1i, _, r1o = ds[i]
    
#     # Decode SELFIES to InChI
#     selfies_str = TOKENIZER.selfies_tokenizer.indexes_to_selfies(r1i.tolist())
#     smiles = selfies.decoder(selfies_str)
#     if not is_valid_smiles(smiles):
#         continue

#     mol = Chem.MolFromSmiles(smiles)
#     inchi = Chem.MolToInchi(mol)

#     # Extract property and value
#     ptok, vtok = r1o[1].item(), r1o[2].item()

#     # Generate context/output tokens
#     ctx, out = inchi_pv_to_ctx_out(inchi, ptok, vtok, nprops=nprops, nsamples=nsamples)

#     # Compute prediction
#     x = r1i.unsqueeze(0).repeat(nsamples, 1).to(DEVICE)
#     ctx = ctx.to(DEVICE)
#     lastidx = (out[0] != TOKENIZER.PAD_IDX).nonzero(as_tuple=True)[0][-2].item()

#     with torch.no_grad():
#         logits = MODEL(x, ctx)[:, lastidx, VALUE_TOKENS]
#         probs = F.softmax(logits, dim=-1)
#         mean_prob_of_1 = probs[:, 1].mean().item()

#     records.append({
#         "inchi": inchi,
#         "property_token": ptok,
#         "value_token": vtok,
#         "prob_of_1": mean_prob_of_1
#     })

# df = pd.DataFrame(records)
# df['value'] = df.apply(lambda row: VALUE_TO_LABEL[row['value_token']], axis=1)

# auc = roc_auc_score(df["value"], df["prob_of_1"])
# print(f"AUC: {auc:.4f}")