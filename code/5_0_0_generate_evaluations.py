"""
rm -rf cache/generate_evaluations/
for SPLIT in {0..5}; do
  LOGDIR="cache/generate_evaluations/${SPLIT}/logs"
  mkdir -p "$LOGDIR"

  SPLIT=$SPLIT LOGDIR="$LOGDIR" PYTHONPATH=./ CUDA_LAUNCH_BLOCKING=1 \
  NCCL_DEBUG=INFO NCCL_ASYNC_ERROR_HANDLING=1 NCCL_IB_DISABLE=1 \
  torchrun --nproc_per_node=8 --nnodes=1 \
    --node_rank=0 code/5_0_0_generate_evaluations.py \
    2> cache/generate_evaluations/$SPLIT/logs/err.log
done

"""
import os, uuid
import torch
import pandas as pd
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pathlib

import cvae.models.multitask_encoder as mte
import cvae.models.multitask_transformer as mt
import cvae.models.mixture_experts as me
import shutil
import logging
from cvae.tokenizer import SelfiesPropertyValTokenizer
from cvae.models.datasets.inmemory_target_property_values_dataset import (
    PreloadedTargetPropertyValuesDataset, TargetPropertyValuesWrapper, MultiTargetPropertyValuesWrapper, MultiTaskPropertySamplesDataset
)
import time
import random
import hashlib
from code.helper.utils import find_optimal_batch_size

# Helper function to convert to hashable types
def _to_hashable(x):
    try:
        if isinstance(x, (list, tuple)):
            return tuple(x)
        if hasattr(x, 'tolist') and not isinstance(x, (str, bytes)):
            return tuple(x.tolist())
    except Exception:
        pass
    return x

# SETUP =================================================================================
SPLIT = os.getenv("SPLIT")
outdir = pathlib.Path(f"cache/generate_evaluations/{SPLIT}")
outdir.mkdir(parents=True, exist_ok=True)

parquetdir = outdir / "evaluations.parquet"
shutil.rmtree(parquetdir, ignore_errors=True) 
parquetdir.mkdir(parents=True, exist_ok=True)

dist.init_process_group("nccl")
local_rank = int(os.environ["LOCAL_RANK"])
torch.cuda.set_device(local_rank)
device = torch.device(f"cuda:{local_rank}")
rank = dist.get_rank()
world_size = dist.get_world_size()

# Only rank 0 creates shared directories
if rank == 0:
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)
dist.barrier()

logdir = outdir / "logs"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=logdir / f"rank_{rank}.log",
    filemode="w"
)

logging.info(f"Rank {rank} started evaluation process.")

model = mte.MultitaskEncoder.load(f"cache/train_multitask_transformer_parallel/logs/split_{SPLIT}/models/me_roundrobin_property_dropout_V3/final_checkpoint").to(device).eval()
tokenizer: SelfiesPropertyValTokenizer = model.tokenizer

eval_props_df = pd.read_parquet("cache/get_benchmark_properties/benchmark_properties.parquet")
eval_props = eval_props_df['property_token'].unique()
eval_props = sorted(eval_props)

# GENERATE EVALUATIONS =================================================================================
logging.info(f"Rank {rank} processing properties: {len(eval_props)}")
preloaded_dataset = PreloadedTargetPropertyValuesDataset(f"cache/build_tensordataset/bootstrap/split_{SPLIT}/test", tokenizer=tokenizer)

logging.info(f"Rank {rank}: Preloaded dataset has {len(preloaded_dataset.samples)} samples")
logging.info(f"Rank {rank}: Properties in dataset: {list(preloaded_dataset.property_to_value_to_sample_idxs.keys())[:10]}...")
logging.info(f"Rank {rank}: Properties to evaluate: {eval_props}")

# START EVALUATION ==================================================================================
nprops_list = [1,2,3,4,5]
minprops = [1,2,3,4,5]

# Pair and shuffle (nprops, prop) for load balancing
tasks = [(n, p) for n in nprops_list for p in eval_props]
random.seed(rank)  # Ensure different shuffle per rank
random.shuffle(tasks)

# Build flattened task list: tuples of (nprops, property_token, minprops)
task_tuples = [(nprops, prop, minp) for (nprops, prop) in tasks for minp in minprops]
logging.info(f"Rank {rank}: built task_tuples, count={len(task_tuples)}")

# Create a single flattened dataset and DataLoader with DistributedSampler
flat_ds = MultiTaskPropertySamplesDataset(preloaded_dataset, task_tuples, nsamples=10_000, max_nprops=max(nprops_list), seed=rank)
logging.info(f"Rank {rank}: flat dataset size {len(flat_ds)}")
sampler = DistributedSampler(flat_ds, shuffle=True)
logging.info(f"Rank {rank}: DistributedSampler created (num_replicas={sampler.num_replicas}, rank={sampler.rank})")
batch_size = 1024 * 10
loader = DataLoader(flat_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
logging.info(f"Rank {rank}: starting flattened DataLoader with batch_size={batch_size}, num_workers=4, total_batches={len(loader)}")

batch_accum = []
with torch.no_grad():
    for repeat in range(1):
        sampler.set_epoch(repeat)
        
        # Create progress bar for inner loop
        pbar = tqdm(
            loader,
            desc=f"Rank {rank} | Repeat {repeat+1}/1",
            position=rank,
            leave=True,
            ncols=120,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
        )
        
        for batch_idx, batch in enumerate(pbar):
            # batch returns: selfies, properties, values, mask, num_properties, property_token, nprops, minprops
            selfies, properties, values, mask, numprops, prop_tokens, nprops_batch, minprops_batch = batch
            selfies = selfies.to(device)
            properties = properties.to(device)
            values = values.to(device)
            mask = mask.to(device)

            logits = model(selfies, properties, values, mask)  # [B, max_nprops, C]

            batch_size_local = logits.size(0)

            # For each sample, the target is at position (nprops) where nprops is per-sample
            target_indices = (nprops_batch-1).to(device)
            batch_indices = torch.arange(batch_size_local, device=device)

            target_logits = logits[batch_indices, target_indices]
            target_probs = torch.softmax(target_logits, dim=1)
            prob_of_1 = target_probs[:, 1]

            true_values = values[batch_indices, target_indices].cpu()

            props_list = properties.view(batch_size_local, -1).cpu().tolist()
            properties_repr = ["-".join(map(str, p)) if isinstance(p, (list, tuple)) else str(p) for p in props_list]

            # Ensure list-like fields are converted to hashable types (tuples)
            chemical_ids = [tuple(x) if isinstance(x, (list, tuple)) else x for x in selfies.view(selfies.shape[0], -1).cpu().tolist()]
            df = pd.DataFrame({
                "chemical_id": chemical_ids,
                "property_token": prop_tokens.cpu().tolist(),
                "properties_repr": properties_repr,
                "prob_of_1": prob_of_1.cpu().tolist(),
                "true_value": true_values.tolist(),
                "numprops": numprops.cpu().tolist(),
                "nprops": nprops_batch.cpu().tolist(),
                "minprops": minprops_batch.cpu().tolist()
            })
            batch_accum.extend(df.to_dict("records"))

            # Update progress bar with additional info
            pbar.set_postfix({
                'Records': len(batch_accum),
                'Batch': batch_size_local
            })

            # Checkpoint saving inside the batch loop
            if len(batch_accum) > 500_000:
                df = pd.DataFrame(batch_accum)
                # Convert list-like and ndarray cells to tuples
                for col in df.columns:
                    df[col] = df[col].apply(_to_hashable)

                df = df.drop_duplicates()
                path = f"{parquetdir}/partial_rank{rank}_{uuid.uuid4()}.parquet"
                df.to_parquet(path, index=False)
                batch_accum = []
                logging.info(f"Rank {rank}: Saved {len(df)} records to {path}")

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                logging.info(f"Rank {rank}: processed flattened batch {batch_idx}/{len(loader)} (accum={len(batch_accum)})")

        pbar.close()

# Final save
if batch_accum:
    logging.info(f"Rank {rank} finalizing results, total records: {len(batch_accum)}")
    df = pd.DataFrame(batch_accum)
    logging.info(f"Rank {rank}: final batch dataframe size before dedup: {len(df)}")
    
    # Convert to hashable and drop duplicates
    for col in df.columns:
        df[col] = df[col].apply(_to_hashable)

    df = df.drop_duplicates()
    path = f"{parquetdir}/final_rank{rank}_{uuid.uuid4()}.parquet"
    df.to_parquet(path, index=False)

dist.barrier()
dist.destroy_process_group()
logging.info(f"Rank {rank} finished evaluation process.")