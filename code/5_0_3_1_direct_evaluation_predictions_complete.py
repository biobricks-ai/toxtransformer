# PYTHONPATH=./ torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 --master_addr=127.0.0.1 --master_port=29500 code/5_0_3_1_direct_evaluation_predictions_complete.py 2> err.log
import os
import torch
import torch.nn.functional as F
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import logging
import torch.distributed as dist
import re
import random
import uuid
import shutil
import math
import pickle
from multiprocessing import Pool, cpu_count
from torch.distributed import init_process_group
from datetime import timedelta
from cvae.models.mixture_experts import MoE

# --- SETTINGS ---
BATCH_SIZE = 1000
SAVE_EVERY_N = 1_000_000
random_seed = 42

# --- Distributed setup (with long timeout) ---
init_process_group(backend="nccl", timeout=timedelta(hours=12))
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)
DEVICE = torch.device(f"cuda:{rank}")
print(f"Rank: {rank} ready.")

# --- Paths ---
cachedir = Path("cache/direct_eval_predictions")
meta_cache_path = cachedir / "meta_by_shard.pkl"
logdir = cachedir / "log"
predictions_dir = cachedir / "predictions_parquet"
shard_dir = Path("cache/direct_eval/hldout_eval_tensors")

# --- Logging ---
logdir.mkdir(parents=True, exist_ok=True)
log_path = logdir / f"incomplete_predict_from_shards_rank{rank}.log"
logging.basicConfig(
    level=logging.INFO,
    filename=log_path,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode='w'
)

# --- Load model ---
MODEL = MoE.load("cache/train_multitask_transformer_parallel/models/moe").to(DEVICE).eval()
TOKENIZER = MODEL.tokenizer
VALUE_TOKEN_MAP = TOKENIZER.value_indexes()
VALUE_TO_LABEL = {v: k for k, v in VALUE_TOKEN_MAP.items()}
VALUE_TOKENS = [VALUE_TOKEN_MAP[0], VALUE_TOKEN_MAP[1]]

# --- Rank 0: build and cache meta_by_shard ---
if rank == 0 and not os.path.exists(meta_cache_path):
    # Load predictions
    predicted_set = set()
    for pf in tqdm(list(predictions_dir.glob("*.parquet")), desc="Loading predictions"):
        df = pd.read_parquet(pf, columns=["inchi", "property_token", "nprops"])
        predicted_set.update(zip(df["inchi"], df["property_token"], df["nprops"]))
    logging.info(f"Rank 0: {len(predicted_set)} predictions loaded for skipping.")

    # Prepare shard list
    shard_paths = sorted(shard_dir.rglob("*.pt"))

    def process_shard(shard_path):
        meta = torch.load(shard_path)["meta"]
        inchi, nprops, ptok, vtok = meta[-1]
        if (inchi, ptok, nprops) in predicted_set:
            return None
        return str(shard_path), meta

    with Pool(32) as pool:
        results = list(tqdm(pool.imap(process_shard, shard_paths), total=len(shard_paths), desc="Indexing shard metadata"))

    res2 = [r for r in results if r is not None]
    meta_by_shard = {Path(k): v for k, v in res2}

    with open(meta_cache_path, "wb") as f:
        pickle.dump(meta_by_shard, f)
    logging.info(f"Rank 0: Indexed {len(meta_by_shard)} shards.")

# --- Barrier and load meta_by_shard ---
dist.barrier()
with open(meta_cache_path, "rb") as f:
    meta_by_shard = pickle.load(f)

# --- Distribute work ---
shard_paths = list(meta_by_shard.keys())
random.Random(random_seed).shuffle(shard_paths)
shard_paths = shard_paths[rank::world_size]

records = []
shard_iter = tqdm(shard_paths, desc=f"Rank {rank}") if rank == 0 else shard_paths

if rank == 0:
    logging.info(f"Rank {rank}: {len(shard_paths)} shards to process.")

for shard_path in shard_iter:
    data = torch.load(shard_path)
    batch_inp = data["inp"]
    batch_ctx = data["ctx"].squeeze(1)
    batch_out = data["out"].squeeze(1)
    metadata = data["meta"]

    match = re.search(r"nprops_(\d+)", shard_path.name)
    path_props = int(match.group(1)) if match else None
    effective_batch_size = int(BATCH_SIZE // math.sqrt(path_props or 1))
    logging.info(f"Batch size: {BATCH_SIZE}, Effective batch size: {effective_batch_size}")

    for i in range(0, len(batch_inp), effective_batch_size):
        x_batch = batch_inp[i:i+effective_batch_size].to(DEVICE)
        ctx_batch = batch_ctx[i:i+effective_batch_size].to(DEVICE)
        ctx_batch = ctx_batch.unsqueeze(1) if ctx_batch.dim() == 2 else ctx_batch
        ctx_batch = ctx_batch[:, 0, :]

        out_batch = batch_out[i:i+effective_batch_size].to(DEVICE)
        out_batch = out_batch.unsqueeze(1) if out_batch.dim() == 2 else out_batch
        out_batch = out_batch[:, 0, :]

        meta_batch = metadata[i:i+effective_batch_size]

        with torch.no_grad():
            logits = MODEL(x_batch, ctx_batch)
            last_indices = [
                (out != TOKENIZER.PAD_IDX).nonzero(as_tuple=True)[0][-2].item()
                for out in out_batch
            ]
            logits = logits[torch.arange(len(logits)), last_indices][:, VALUE_TOKENS]
            probs = F.softmax(logits, dim=-1)
            prob_of_1 = probs[:, 1].cpu().tolist()

        for j in range(len(meta_batch)):
            inchi, nprops, ptok, vtok = meta_batch[j]
            records.append({
                "inchi": inchi,
                "property_token": ptok,
                "nprops": nprops,
                "value_token": vtok,
                "prob_of_1": prob_of_1[j]
            })

        if len(records) >= SAVE_EVERY_N:
            df_chunk = pd.DataFrame(records)
            df_chunk["value"] = df_chunk["value_token"].map(VALUE_TO_LABEL)
            uuid_path = predictions_dir / f"{uuid.uuid4()}.parquet"
            df_chunk.to_parquet(uuid_path, index=False)
            logging.info(f"📝 Rank {rank} wrote intermediate predictions to {uuid_path}")
            records.clear()

# --- Final save ---
if records:
    df = pd.DataFrame(records)
    df["value"] = df["value_token"].map(VALUE_TO_LABEL)
    uuid_path = predictions_dir / f"{uuid.uuid4()}.parquet"
    df.to_parquet(uuid_path, index=False)
    logging.info(f"✅ Rank {rank} wrote final predictions to {uuid_path}")

# --- Shutdown ---
dist.barrier()
dist.destroy_process_group()