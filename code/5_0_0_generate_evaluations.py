#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHONPATH=./ torchrun --standalone --nproc-per-node=8 --master-port=29500 code/generate_evaluations.py 2> cache/generate_evaluations/logs/err.log
# TODO this is broken and currently not in use. Either remove or fix.

import os, uuid, pathlib, shutil, logging, random
import pandas as pd, torch, numpy as np, tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from itertools import product
import faulthandler
faulthandler.enable()

from cvae.tokenizer import SelfiesPropertyValTokenizer
import cvae.models.mixture_experts as me
import cvae.models.datasets.restricted_dataset as rd

torch._nested_tensor_from_mask_left = None

class EvalContext:
    def __init__(self, rank, local_rank, model, tokenizer, device, nprops, batch_size):
        self.rank = rank
        self.local_rank = local_rank
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.nprops = nprops
        self.batch_size = batch_size

def hash_input(tensor, pad_idx):
    trimmed = tensor[tensor != pad_idx].tolist()
    return "_".join(map(str, trimmed))

def run_eval(i, raw_inp, raw_teach, raw_out, context):
    inp, teach, out = raw_inp.to(context.device), raw_teach.to(context.device), raw_out.to(context.device)
    valid = torch.sum(torch.isin(out, context.tokenizer.value_indexes_tensor), dim=1) >= context.nprops
    inp, teach, out = inp[valid], teach[valid], out[valid, 1:(2 * context.nprops + 1)]
    if inp.shape[0] == 0: return pd.DataFrame()
    input_hashes = [hash_input(row, context.tokenizer.PAD_IDX) for row in inp]
    
    assert inp.shape[0] == teach.shape[0] == out.shape[0] != 0, f"Rank {context.rank} - {inp.shape[0]} inputs → {valid.sum().item()} valid after token filtering"
    logging.info(f"[Rank {context.rank}] {inp.shape[0]} inputs → {valid.sum().item()} valid after token filtering")
    
    with torch.no_grad():
        prob = context.model(inp, teach)
        if prob is None or torch.isnan(prob).any() or torch.isinf(prob).any():
            return pd.DataFrame()
        prob = torch.softmax(prob, dim=2)

    assay_pos = torch.arange(0, context.nprops * 2, 2, device=out.device)
    value_pos = assay_pos + 1
    assays, values = out[:, assay_pos], out[:, value_pos]
    prob_vals = torch.argmax(prob, dim=2)[:, value_pos]
    rawprobs = prob[:, value_pos, :][:, :, context.tokenizer.value_indexes_tensor]
    probs = (rawprobs / rawprobs.sum(dim=2, keepdim=True))[:, :, 1]

    assays_np = assays.cpu().numpy().astype(str).flatten()
    values_np = values.cpu().numpy().astype(str).flatten()
    prob_vals_np = prob_vals.cpu().numpy().flatten()
    probs_np = probs.cpu().numpy().flatten()
    chemical_id_np = np.repeat(input_hashes, context.nprops)
    position = np.tile(np.arange(context.nprops), len(input_hashes))

    assays_str = assays.cpu().numpy().astype(str)
    values_str = values.cpu().numpy().astype(str)
    prior_assays = [' + '.join(row[:j+1]) for row in assays_str for j in range(context.nprops)]
    prior_values = [' + '.join(row[:j+1]) for row in values_str for j in range(context.nprops)]

    return pd.DataFrame({
        'batch': i,
        'chemical_id': chemical_id_np,
        'prior_assays': prior_assays,
        'prior_values': prior_values,
        'assay': assays_np,
        'value': values_np,
        'probs': probs_np,
        'nprops': position,
        'prob_assays': assays_np,
        'prob_vals': prob_vals_np
    })

def setup(rank):
    outdir = pathlib.Path("cache/generate_evaluations")
    tmpdir = outdir / "temp"
    logdir = outdir / "logs"
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = logdir / f"log_{rank}.txt"
    log_format = '%(asctime)s - RANK {} - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'.format(rank)
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S', force=True, filemode='w')
    logging.info("--- Starting Setup ---")
    if rank == 0:
        outdir.mkdir(exist_ok=True, parents=True)
        shutil.rmtree(tmpdir, ignore_errors=True)
        tmpdir.mkdir(exist_ok=True, parents=True)
    return outdir, tmpdir

def flush_to_parquet(accum, tmpdir, rank, label):
    if accum:
        df = pd.concat(accum, ignore_index=True)
        path = tmpdir / f"predictions_{label}_r{rank}_{uuid.uuid4().hex[:8]}.parquet"
        df.to_parquet(path, index=False)

def main_worker(context, repetitions, outdir, tmpdir):
    bp = pd.read_parquet("cache/get_benchmark_properties/benchmark_properties.parquet")
    props = list(set(bp['property_token']))
    dataset = rd.FastPropertyGuaranteeDataset(
        path="cache/build_tensordataset/multitask_tensors/hld",
        tokenizer=context.tokenizer,
        nprops=context.nprops,
        target_props=props,
        target_positions=[0, 4],
        sampling_weights={},
        skip_small_nprops=True
    )

    # sampler = DistributedSampler(dataset, dist.get_world_size(), context.rank, shuffle=False)
    loader = DataLoader(
        dataset,
        batch_size=context.batch_size,
        num_workers=4,  # or adjust to CPU cores / num ranks
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=False
    )

    accum, row_count = [], 0
    threshold = 250_000

    for rep in range(repetitions):
        logging.info(f"Rank {context.rank} - Repeat {rep+1}/{repetitions} - current rows: {row_count // 1000}k")
        random.shuffle(props)

        for prop in props:
            logging.info(f"Rank {context.rank} - Setting sampling weights for prop {prop}")
            if not dataset.has_property(prop):
                continue

            dataset.set_sampling_weights({prop: 1.0})
            # sampler.set_epoch(rep)
            sampled = 0

            for i, (inp, teach, out) in enumerate(loader):
                df = run_eval(i, inp, teach, out, context)
                logging.info(f"Rank {context.rank} - Evaluated {i} rows {len(df)} rows, accum {len(accum)} rows")
                if not df.empty:
                    accum.append(df)
                    row_count += len(df)
                    sampled += inp.size(0)

                if row_count >= threshold:
                    flush_to_parquet(accum, tmpdir, context.rank, f"rep{rep}_prop{prop}")
                    accum, row_count = [], 0

                if sampled >= 10_000:
                    break

        flush_to_parquet(accum, tmpdir, context.rank, "final")
    flush_to_parquet(accum, tmpdir, context.rank, "final")


if __name__ == "__main__":
    repetitions = 10000
    rank = int(os.environ.get("RANK", -1))
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    outdir, tmpdir = setup(rank)

    try:
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        model = me.MoE.load("cache/train_multitask_transformer_parallel/models/moe").to(device)
        model = torch.compile(model,  mode="reduce-overhead", fullgraph=False)
        model.eval()

        tokenizer: SelfiesPropertyValTokenizer = model.tokenizer
        tokenizer.assay_indexes_tensor = torch.tensor(list(tokenizer.assay_indexes().values()), device=device)
        tokenizer.value_indexes_tensor = torch.tensor(list(tokenizer.value_indexes().values()), device=device)

        context = EvalContext(rank, local_rank, model, tokenizer, device, nprops=5, batch_size=128)

        dist.barrier()
        main_worker(context, repetitions, outdir, tmpdir)
        dist.barrier()

    except Exception as e:
        logging.exception(f"Unhandled Exception (Rank {rank})")

    finally:
        if dist.is_initialized():
            dist.destroy_process_group()
