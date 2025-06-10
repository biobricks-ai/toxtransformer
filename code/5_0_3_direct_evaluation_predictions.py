# PYTHONPATH=./ python code/5_0_3_direct_evaluation_predictions.py

import torch
import torch.nn.functional as F
import pandas as pd
import pickle
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
import logging

from cvae.models.mixture_experts import MoE

# --- Logging setup ---
cachedir = Path("cache/direct_eval_predictions")
cachedir.mkdir(parents=True, exist_ok=True)
logdir = cachedir / "log"
logdir.mkdir(parents=True, exist_ok=True)
log_path = logdir / "predict_from_shards.log"

logging.basicConfig(
    level=logging.INFO,
    filename=log_path,
    format="%(asctime)s [%(levelname)s] %(message)s",
    filemode='w'
)
logging.getLogger().addHandler(logging.StreamHandler())

# --- Setup ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL = MoE.load("cache/train_multitask_transformer_parallel/models/moe").to(DEVICE).eval()
TOKENIZER = MODEL.tokenizer
VALUE_TOKEN_MAP = TOKENIZER.value_indexes()
VALUE_TO_LABEL = {v: k for k, v in VALUE_TOKEN_MAP.items()}
VALUE_TOKENS = [VALUE_TOKEN_MAP[0], VALUE_TOKEN_MAP[1]]

# --- Load Shards ---
shard_dir = Path("cache/direct_eval/hldout_eval_tensors")
shard_paths = sorted(shard_dir.rglob("*.pt"))
# shard_path = [p for p in shard_paths if 'nprops_10.pt' in p.name][0]
# shard_path = [p for p in shard_paths if 'nprops_1.pt' in p.name][0]
records = []
total_count = 0
BATCH_SIZE = 512

# predictions
for shard_path in tqdm(shard_paths, desc="Processing shards"):
    # logging.info(f"Processing shard: {shard_path.name}")
    data = torch.load(shard_path)
    batch_inp = data["inp"]
    batch_ctx = data["ctx"].squeeze(1)
    batch_out = data["out"].squeeze(1)
    metadata = data["meta"]

    for i in range(0, len(batch_inp), BATCH_SIZE):
        x_batch = batch_inp[i:i+BATCH_SIZE].to(DEVICE)   # nprops_1 28, 120, nprops_10 28, 120
        
        ctx_batch = batch_ctx[i:i+BATCH_SIZE].to(DEVICE) 
        ctx_batch = ctx_batch.unsqueeze(1) if ctx_batch.dim() == 2 else ctx_batch
        ctx_batch = ctx_batch[:,0,:]
        
        out_batch = batch_out[i:i+BATCH_SIZE].to(DEVICE) # nprops_1 28, 4
        out_batch = out_batch.unsqueeze(1) if out_batch.dim() == 2 else out_batch
        out_batch = out_batch[:,0,:]
        
        meta_batch = metadata[i:i+BATCH_SIZE]

        with torch.no_grad():
            logits = MODEL(x_batch, ctx_batch)

            last_indices = []
            for out in out_batch:
                nonpad = (out != TOKENIZER.PAD_IDX).nonzero(as_tuple=True)[0]
                last_idx = nonpad[-2].item()
                last_indices.append(last_idx)

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


# --- Save & Analyze ---
df = pd.DataFrame(records)
df["value"] = df["value_token"].map(VALUE_TO_LABEL)

# calculate AUC
auc = roc_auc_score(df["value"], df["prob_of_1"])
logging.info(f"AUC: {auc:.4f}")

# Save predictions
outpath = cachedir / "predictions.parquet"
df.to_parquet(outpath, index=False)
logging.info(f"✅ Saved predictions to {outpath}")