# PYTHONPATH=./ torchrun --nproc_per_node=8 --nnodes=1 --node_rank=0 code/5_0_0_generate_evaluations.py 2> cache/generate_evaluations/logs/err.log
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

# SETUP =================================================================================
outdir = pathlib.Path("cache/generate_evaluations/")
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


logdir = pathlib.Path("cache/generate_evaluations/logs")
logdir.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    filename=logdir / f"rank_{rank}.log",
    filemode="w"
)

logging.info(f"Rank {rank} started evaluation process.")
model = mte.MultitaskEncoder.load("cache/train_multitask_transformer_parallel/models/mt/best_loss").to(device).eval()
tokenizer: SelfiesPropertyValTokenizer = model.tokenizer

eval_props_df = pd.read_parquet("cache/get_benchmark_properties/benchmark_properties.parquet")
eval_props_df = eval_props_df.query('source == "BBBP"')
eval_props = eval_props_df['property_token'].unique()
eval_props = sorted(eval_props)

# GENERATE EVALUATIONS =================================================================================
logging.info(f"Rank {rank} processing properties: {len(eval_props)}")
preloaded_dataset = PreloadedTargetPropertyValuesDataset("cache/build_tensordataset/multitask_tensors/hld", tokenizer=tokenizer)

# --- Ground truth for prop 5320 ---
prop_token_gt = 5320
value_to_idxs_gt = preloaded_dataset.property_to_value_to_sample_idxs.get(int(prop_token_gt), {})
gt_idxs = set()
for _, idxs in value_to_idxs_gt.items():
    for idx in idxs:
        if preloaded_dataset.sample_to_numprops.get(idx, 0) >= 1:  # minprops=1
            gt_idxs.add(int(idx))
logging.info(f"GT count for prop {prop_token_gt}: {len(gt_idxs)} (expect 147)")

# Save to disk once (rank 0)
if rank == 0:
    import json
    with open(outdir / "gt_prop5320_baseidx.json", "w") as f:
        json.dump(sorted(gt_idxs), f)

# Generate a log of the number of samples per property_token
# Debug: Check what's in the preloaded dataset
logging.info(f"Rank {rank}: Preloaded dataset has {len(preloaded_dataset.samples)} samples")
logging.info(f"Rank {rank}: Properties in dataset: {list(preloaded_dataset.property_to_value_to_sample_idxs.keys())[:10]}...")  # Show first 10
logging.info(f"Rank {rank}: Properties to evaluate: {eval_props}")

# TEST LOGS ==================================================================================
# Generate a log of the number of samples per property_token
property_sample_counts = {}
for prop_token in eval_props:
    # Get all sample indices that contain this property
    value_to_idxs = preloaded_dataset.property_to_value_to_sample_idxs.get(prop_token, {})
    total_samples = sum(len(idxs) for idxs in value_to_idxs.values())
    
    # Also count samples with minimum properties requirement
    samples_with_minprops = {}
    for minprops in [1, 2, 3, 4, 5]:
        count = 0
        for value, idxs in value_to_idxs.items():
            for idx in idxs:
                if preloaded_dataset.sample_to_numprops.get(idx, 0) >= minprops:
                    count += 1
        samples_with_minprops[minprops] = count
    
    property_sample_counts[prop_token] = {
        'total_samples': total_samples,
        'samples_by_minprops': samples_with_minprops,
        'unique_values': len(value_to_idxs),
        'value_counts': {value: len(idxs) for value, idxs in value_to_idxs.items()},
    }

# Log the results
logging.info(f"Rank {rank} - Sample counts per property:")
for prop_token, counts in property_sample_counts.items():
    logging.info(f"  Property {prop_token}: {counts['total_samples']} total samples, "
                f"{counts['unique_values']} unique values")
    
    # Log overall value distribution
    logging.info(f"    Value distribution (total):")
    for value, count in counts['value_counts'].items():
        logging.info(f"      Value {value}: {count} samples")
    
    # Log samples by minprops requirement
    for minprops, count in counts['samples_by_minprops'].items():
        logging.info(f"    >= {minprops} props: {count} samples")

# Save detailed counts to file for analysis
if rank == 0:  # Only save once from rank 0
    import json
    counts_path = outdir / "property_sample_counts.json"
    with open(counts_path, 'w') as f:
        # Convert int keys to strings for JSON serialization
        json_counts = {str(k): v for k, v in property_sample_counts.items()}
        json.dump(json_counts, f, indent=2)
    logging.info(f"Saved detailed sample counts to {counts_path}")

# START EVALUATION ==================================================================================
nprops_list = [1,2,3,4,5]

# Pair and shuffle (nprops, prop) for load balancing
tasks = [(n, p) for n in nprops_list for p in eval_props]
random.seed(rank)  # Ensure different shuffle per rank
random.shuffle(tasks)

# import numpy as np; prop = np.int64(14044); nprops = 5
minprops = [1,2,3,4,5]
minprop_prop_iter = list(enumerate((minprop, task) for task in tasks for minprop in minprops))

total_tasks = len(minprop_prop_iter)
start_time = time.time()
logging.info(f"Rank {rank} has {total_tasks} tasks to process.")

# Build flattened task list: tuples of (nprops, property_token, minprops)
task_tuples = [(nprops, prop, minp) for (minp, (nprops, prop)) in (t[1] for t in minprop_prop_iter)]
logging.info(f"Rank {rank}: built task_tuples, count={len(task_tuples)}")

# Create a single flattened dataset and DataLoader with DistributedSampler
flat_ds = MultiTaskPropertySamplesDataset(preloaded_dataset, task_tuples, nsamples=10_000, max_nprops=max(nprops_list), seed=rank)
logging.info(f"Rank {rank}: flat dataset size {len(flat_ds)}")
sampler = DistributedSampler(flat_ds, shuffle=True)
logging.info(f"Rank {rank}: DistributedSampler created (num_replicas={sampler.num_replicas}, rank={sampler.rank})")
batch_size = 1024 * 100
loader = DataLoader(flat_ds, batch_size=batch_size, sampler=sampler, num_workers=4, pin_memory=True)
logging.info(f"Rank {rank}: starting flattened DataLoader with batch_size={batch_size}, num_workers=4, total_batches={len(loader)}")

batch_accum = []
with torch.no_grad():
    for repeat in tqdm(range(1000)):
        sampler.set_epoch(repeat)
        for batch_idx, batch in enumerate(loader):
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

            # Ensure list-like fields are converted to hashable types (tuples) so
            # that drop_duplicates() works without TypeError: unhashable type: 'list'.
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

            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
                logging.info(f"Rank {rank}: processed flattened batch {batch_idx}/{len(loader)} (accum={len(batch_accum)})")

        if len(batch_accum) > 500_000:
            df = pd.DataFrame(batch_accum)
            # Convert list-like and ndarray cells to tuples so drop_duplicates can hash them
            def _to_hashable(x):
                try:
                    if isinstance(x, (list, tuple)):
                        return tuple(x)
                    if hasattr(x, 'tolist') and not isinstance(x, (str, bytes)):
                        return tuple(x.tolist())
                except Exception:
                    pass
                return x

            for col in df.columns:
                df[col] = df[col].apply(_to_hashable)

            df = df.drop_duplicates()
            path = f"{parquetdir}/partial_rank{rank}_{uuid.uuid4()}.parquet"
            df.to_parquet(path, index=False)
            batch_accum = []
            logging.info(f"Rank {rank}: Saved {len(df)} records to {path}")

if batch_accum:
    logging.info(f"Rank {rank} finalizing results, total records: {len(batch_accum)}")
    df = pd.DataFrame(batch_accum)
    # Drop duplicates before writing the final chunk for this rank
    def _to_hashable(x):
        try:
            if isinstance(x, (list, tuple)):
                return tuple(x)
            if hasattr(x, 'tolist') and not isinstance(x, (str, bytes)):
                return tuple(x.tolist())
        except Exception:
            pass
        return x

    for col in df.columns:
        df[col] = df[col].apply(_to_hashable)

    df = df.drop_duplicates()
    path = f"{parquetdir}/final_rank{rank}_{uuid.uuid4()}.parquet"
    df.to_parquet(path, index=False)

dist.barrier()
dist.destroy_process_group()
logging.info(f"Rank {rank} finished evaluation process.")

import pandas as pd
df1 = pd.read_parquet('cache/generate_evaluations/evaluations.parquet')
chem_repr = df1['chemical_id'].apply(lambda x: '-'.join(map(str, x)) if isinstance(x, (list, tuple)) else str(x))
df1['chem_repr'] = chem_repr
df2 = df1[['chem_repr','nprops','property_token','true_value']].drop_duplicates()
df1[['nprops','property_token','true_value']].value_counts()\
    .sort_index(level=['nprops', 'true_value'])