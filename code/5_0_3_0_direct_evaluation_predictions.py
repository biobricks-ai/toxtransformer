import os
import torch
import torch.nn.functional as F
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import logging
import torch.distributed as dist
import re
import random
import uuid
import shutil
from cvae.models.mixture_experts import MoE
import math

# --- Distributed setup ---
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
print(f"Rank: {rank}")
world_size = dist.get_world_size()
torch.cuda.set_device(rank)
DEVICE = torch.device(f"cuda:{rank}")

# --- clear cache ---
if rank == 0:
    shutil.rmtree("cache/direct_eval_predictions/predictions_parquet", ignore_errors=True)
os.makedirs("cache/direct_eval_predictions/predictions_parquet", exist_ok=True)

dist.barrier()
# --- Logging setup ---
cachedir = Path("cache/direct_eval_predictions")
cachedir.mkdir(parents=True, exist_ok=True)
logdir = cachedir / "log"
logdir.mkdir(parents=True, exist_ok=True)
log_path = logdir / f"predict_from_shards_rank{rank}.log"

logging.basicConfig(
    level=logging.INFO,
    filename=log_path,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode='w'
)

# --- Model ---
MODEL = MoE.load("cache/train_multitask_transformer_parallel/models/moe").to(DEVICE).eval()
TOKENIZER = MODEL.tokenizer
VALUE_TOKEN_MAP = TOKENIZER.value_indexes()
VALUE_TO_LABEL = {v: k for k, v in VALUE_TOKEN_MAP.items()}
VALUE_TOKENS = [VALUE_TOKEN_MAP[0], VALUE_TOKEN_MAP[1]]

# --- Load Shards ---
shard_dir = Path("cache/direct_eval/hldout_eval_tensors")
shard_paths = sorted(shard_dir.rglob("*.pt"))
random_seed = 42
random.Random(random_seed).shuffle(shard_paths)
shard_paths = shard_paths[rank::world_size]  # split across ranks

BATCH_SIZE = 1000
SAVE_EVERY_N = 1e6
records = []

dist.barrier()
shard_iter = shard_paths if rank != 0 else tqdm(shard_paths, desc=f"Rank {rank} processing shards")
for shard_path in shard_iter:
    data = torch.load(shard_path)
    batch_inp = data["inp"]
    batch_ctx = data["ctx"].squeeze(1)
    batch_out = data["out"].squeeze(1)
    metadata = data["meta"]

    match = re.search(r"nprops_(\d+)", shard_path.name)
    path_props = int(match.group(1)) if match else None

    effective_batch_size = int(BATCH_SIZE // math.sqrt(path_props))
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
            last_indices = [(out != TOKENIZER.PAD_IDX).nonzero(as_tuple=True)[0][-2].item() for out in out_batch]
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
                uuid_path = cachedir / "predictions_parquet" / f"{uuid.uuid4()}.parquet"
                uuid_path.parent.mkdir(parents=True, exist_ok=True)
                df_chunk.to_parquet(uuid_path, index=False)
                logging.info(f"📝 Rank {rank} wrote intermediate predictions to {uuid_path}")
                records.clear()

# Save any remaining records
if records:
    df = pd.DataFrame(records)
    df["value"] = df["value_token"].map(VALUE_TO_LABEL)
    uuid_path = cachedir / "predictions_parquet" / f"{uuid.uuid4()}.parquet"
    uuid_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(uuid_path, index=False)
    logging.info(f"✅ Rank {rank} wrote final predictions to {uuid_path}")

# Clean shutdown
dist.barrier()
dist.destroy_process_group()

