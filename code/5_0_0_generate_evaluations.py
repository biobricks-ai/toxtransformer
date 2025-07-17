# PYTHONPATH=./ torchrun --nproc_per_node=2 --nnodes=1 --node_rank=0 code/5_0_0_generate_evaluations.py
import os, uuid
import torch
import pandas as pd
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import pathlib

import cvae.models.mixture_experts as me
import shutil
import logging
from cvae.tokenizer import SelfiesPropertyValTokenizer
from cvae.models.datasets.inmemory_sequence_shift_dataset import (
    PreloadedSequenceShiftDataset, TargetPropertySequenceShiftWrapper
)
import time
import random
import hashlib

outdir = pathlib.Path("cache/generate_evaluations/")
shutil.rmtree(outdir, ignore_errors=True)  # clear previous results
outdir.mkdir(parents=True, exist_ok=True)

parquetdir = outdir / "evaluations.parquet"
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
model = me.MoE.load("cache/train_multitask_transformer_parallel/models/moe").to(device).eval()

tokenizer: SelfiesPropertyValTokenizer = model.tokenizer
tokenizer.assay_indexes_tensor = torch.tensor(list(tokenizer.assay_indexes().values()), device=device)
tokenizer.value_indexes_tensor = torch.tensor(list(tokenizer.value_indexes().values()), device=device)
valueindex = tokenizer.value_indexes()
valuemap = {v: k for k, v in valueindex.items()}  # inverse map

eval_props_df = pd.read_parquet("cache/get_benchmark_properties/benchmark_properties.parquet")
eval_props = eval_props_df['property_token'].unique()
eval_props = sorted(eval_props)
eval_props = eval_props[rank::world_size]  # split work across ranks

logging.info(f"Rank {rank} processing properties: {len(eval_props)}")
preloaded_dataset = PreloadedSequenceShiftDataset("cache/build_tensordataset/multitask_tensors/hld", tokenizer=tokenizer)

nprops_list = [1, 2, 3, 4, 5]
batch_accum = []

# Pair and shuffle (nprops, prop) for load balancing
tasks = [(n, p) for n in nprops_list for p in eval_props]
random.seed(rank)  # Ensure different shuffle per rank
random.shuffle(tasks)

total_tasks = len(tasks)
start_time = time.time()
logging.info(f"Rank {rank} has {total_tasks} tasks to process.")

value_tensor = torch.tensor(list(valueindex.values()), device=device)
index_of_1 = list(valueindex.values()).index(valueindex[1])

# import numpy as np; prop = np.int64(14044); nprops = 5

for minprops in range(1, 6):
    for task_idx, (nprops, prop) in tqdm(enumerate(tasks), total=total_tasks):
        ds = TargetPropertySequenceShiftWrapper(preloaded_dataset, nprops=nprops, target_property_token=prop, nsamples=200, minprops=minprops)
        logging.info(f"Rank {rank} processing property={prop}, nprops={nprops} ({task_idx + 1}/{total_tasks}) len(ds): {len(ds)}")
        batch_size = 3000 // nprops
        loader = DataLoader(ds, batch_size=batch_size)
        # batch = next(iter(loader))

        with torch.no_grad():
            for batch in loader:
                inp, teach, out, numprops = (x.to(device) for x in batch)
                prob = model(inp, teach)
                target_val_indices = []
                for seq in out:
                    end = (seq == tokenizer.END_IDX).nonzero(as_tuple=True)[0].item()
                    target_val_indices.append(end - 1)
                
                batch_indices = torch.arange(prob.size(0), device=device)
                value_probs = prob[batch_indices, target_val_indices][:, value_tensor].softmax(dim=1)

                prob_of_1 = value_probs[:, index_of_1]

                true_token = out[batch_indices, target_val_indices].cpu()
                true_val = [valuemap[int(tok)] for tok in true_token]


                df = pd.DataFrame({
                    "chemical_id": inp.view(inp.shape[0], -1).cpu().tolist(),
                    "property_token": [prop] * len(true_val),
                    "prob_of_1": prob_of_1.cpu().tolist(),
                    "true_value": true_val,
                    "numprops": numprops.cpu().tolist(),
                    "nprops": [nprops] * len(true_val),
                    "minprops": [minprops] * len(true_val)
                })
                batch_accum.extend(df.to_dict("records"))

                if len(batch_accum) > 1_000_000:
                    df = pd.DataFrame(batch_accum)
                    path = f"{parquetdir}/partial_rank{rank}_{uuid.uuid4()}.parquet"
                    df.to_parquet(path, index=False)
                    batch_accum = []

        # Log estimated remaining time
        elapsed = time.time() - start_time
        tasks_done = task_idx + 1
        tasks_remaining = total_tasks - tasks_done
        time_per_task = elapsed / tasks_done
        time_remaining = tasks_remaining * time_per_task
        logging.info(f"Rank {rank}: {tasks_remaining} tasks remaining, estimated time left: {time_remaining/60:.2f} min")


if batch_accum:
    df = pd.DataFrame(batch_accum)
    path = f"{parquetdir}/final_rank{rank}_{uuid.uuid4()}.parquet"
    df.to_parquet(path, index=False)

dist.barrier()
dist.destroy_process_group()