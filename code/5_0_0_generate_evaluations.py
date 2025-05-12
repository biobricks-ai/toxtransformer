#!/usr/bin/env python
# -*- coding: utf-8 -*-
# torchrun --standalone --nproc-per-node=8 --master-port=29500 code/5_0_generate_evaluations.py 2> cache/generate_evaluations/logs/err.log

import sys
import argparse
import math
import os, itertools, uuid, pathlib, shutil, logging
import pandas as pd, torch, numpy as np, tqdm
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader

# Enable faulthandler early to catch segfaults better
import faulthandler
faulthandler.enable()

import cvae.tokenizer
import cvae.models.multitask_transformer as mt
import cvae.models.mixture_experts as me
import cvae.utils
from cvae.tokenizer import SelfiesPropertyValTokenizer

import glob
import pyarrow.parquet as pq
import pyarrow.dataset as ds

# Suppress the specific nested tensor warning if it's too noisy (optional)
# import warnings
# warnings.filterwarnings("ignore", message="The PyTorch API of nested tensors is in prototype stage*", category=UserWarning)

torch._nested_tensor_from_mask_left = None

class EvalContext:
    def __init__(self, rank, local_rank, model, tokenizer, device, perm_indices, perm_count, nprops, batch_size):
        self.rank = rank
        self.local_rank = local_rank
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.perm_indices = perm_indices
        self.perm_count = perm_count
        self.nprops = nprops
        self.batch_size = batch_size

import hashlib

def run_eval(i, raw_inp, raw_out, context: EvalContext):
    inp, raw_out = raw_inp.to(context.device), raw_out.to(context.device)

    x = torch.sum(torch.isin(raw_out, context.tokenizer.value_indexes_tensor), dim=1) >= context.nprops
    inp = inp[x]
    trunc_out = raw_out[x, 1:(2 * context.nprops + 1)].reshape(-1, context.nprops, 2)

    if inp.shape[0] == 0:
        return pd.DataFrame()

    # Generate hashes as IDs
    def hash_row(tensor):
        byte_data = tensor.cpu().numpy().tobytes()
        return hashlib.sha1(byte_data).hexdigest()

    input_hashes = [hash_row(row) for row in inp]

    perm_out = torch.cat([trunc_out[:, list(perm), :] for perm in context.perm_indices], dim=0).reshape(-1, context.nprops * 2)
    sep_tensor = torch.full((perm_out.size(0), 1), context.tokenizer.SEP_IDX, device=context.device)
    out = torch.cat([sep_tensor, perm_out, torch.zeros_like(sep_tensor)], dim=1)
    teach = torch.cat([torch.ones_like(sep_tensor), out[:, :-1]], dim=1)
    rep_inp = inp.repeat(context.perm_count, 1)

    with torch.no_grad():
        prob = context.model(rep_inp, teach)
        if prob is None or torch.isnan(prob).any() or torch.isinf(prob).any():
            return pd.DataFrame()
        prob = torch.softmax(prob, dim=2)

    assays_mask = torch.isin(out, context.tokenizer.assay_indexes_tensor)
    assays = out[assays_mask]

    values_mask = torch.isin(out, context.tokenizer.value_indexes_tensor)
    values = out[values_mask]
    prob_vals = torch.argmax(prob, dim=2)[values_mask]
    rawprobs = prob[values_mask][:, context.tokenizer.value_indexes_tensor]
    probs = (rawprobs / rawprobs.sum(dim=1, keepdim=True))[:, 1]

    assays_np, values_np, prob_vals_np, probs_np = map(lambda x: x.cpu().numpy(), [assays, values, prob_vals, probs])

    position = np.tile(np.arange(context.nprops), len(input_hashes) * context.perm_count)
    chemical_id_np = np.repeat(input_hashes, context.perm_count * context.nprops)

    assays_reshaped = assays_np.reshape(-1, context.nprops).astype(str)
    prior_assays = [' + '.join(assays_reshaped[k, :j + 1]) for k in range(len(assays_reshaped)) for j in range(context.nprops)]

    values_reshaped = values_np.reshape(-1, context.nprops).astype(str)
    prior_values = [' + '.join(values_reshaped[k, :j + 1]) for k in range(len(values_reshaped)) for j in range(context.nprops)]

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



def setup(rank): # Pass rank for logging before dist init maybe
    # Basic logging setup first
    outdir = pathlib.Path("cache/generate_evaluations")
    tmpdir = outdir / "temp"
    logdir = outdir / "logs"
    logdir.mkdir(exist_ok=True, parents=True)
    logfile = (logdir / f"log_{rank}.txt").as_posix() # Log using rank passed initially
    log_format = '%(asctime)s - RANK {} - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'.format(rank)
    logging.basicConfig(filename=logfile, level=logging.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S', force=True, filemode='w') # force=True helps if called multiple times?

    logging.info("--- Starting Setup ---")
    logging.info(f"Process ID: {os.getpid()}")
    logging.info(f"Environment RANK: {os.environ.get('RANK')}, LOCAL_RANK: {os.environ.get('LOCAL_RANK')}, WORLD_SIZE: {os.environ.get('WORLD_SIZE')}")

    if rank == 0:
        logging.info("Rank 0 performing initial setup")
        outdir.mkdir(exist_ok=True, parents=True)
        shutil.rmtree(tmpdir, ignore_errors=True)
        tmpdir.mkdir(exist_ok=True, parents=True)
    
    return outdir, tmpdir

def cleanup():
    if dist.is_initialized():
        logging.info(f"Rank {dist.get_rank()} - Destroying process group.")
        dist.destroy_process_group()

def main_worker(context: EvalContext, repetitions, outdir, tmpdir):
    rank = context.rank
    tokenizer = context.tokenizer
    logging.info(f"Rank {rank} - Starting main_worker.")
    logging.info(f"Rank {rank} - Initializing Dataset.")
    dataset = mt.SequenceShiftDataset(
        path="cache/build_tensordataset/multitask_tensors/hld", 
        tokenizer=tokenizer, 
        nprops=5
    )
    sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=rank, shuffle=False)
    dataloader = DataLoader(dataset, batch_size=context.batch_size, sampler=sampler,
        num_workers=4, pin_memory=True, prefetch_factor=5, persistent_workers=True, pin_memory_device=f"cuda:{context.local_rank}") # Specify pin_memory_device

    logging.info(f"Rank {rank} - Initialized DataLoader with {len(dataloader)} batches per epoch.")
    logging.info(f"Rank {rank} - Using batch size: {context.batch_size}, num_workers=4")

    seen_inputs = set() # Consider if this is needed or causing issues
    batch_accum = []
    total_processed_count = 0

    for repeat in range(repetitions):
        logging.info(f"Rank {rank} - Starting repeat {repeat+1}/{repetitions}")
        sampler.set_epoch(repeat) # Important for shuffling with DistributedSampler if shuffle=True
        for i, batch_data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Rank {rank} Repeat {repeat+1}"):

                raw_inp, _, raw_out = batch_data

                batch_tuples = tuple(map(lambda x, y: (tuple(x.tolist()), tuple(y.tolist())), raw_inp, raw_out))
                new_inputs_mask = [t not in seen_inputs for t in batch_tuples]
                seen_inputs.update(batch_tuples)

                if any(new_inputs_mask):
                    new_raw_inp = raw_inp[new_inputs_mask]
                    new_raw_out = raw_out[new_inputs_mask]
                    batch_df = run_eval(i, new_raw_inp, new_raw_out, context)
                    if not batch_df.empty:
                            batch_accum.append(batch_df)
                            total_processed_count += new_raw_inp.shape[0] # Count processed items

                # Check accumulation size and save periodically
                current_accum_rows = sum(len(df) for df in batch_accum)
                if batch_accum and current_accum_rows > 1_000_000: # Save threshold
                    logging.info(f"Rank {rank} - Saving batch accumulation at step {i}, repeat {repeat}. Accumulated rows: {current_accum_rows}")
                    pd.concat(batch_accum).to_parquet(tmpdir / f"multitask_predictions_{rank}_{repeat}_{i}_{uuid.uuid4()}.parquet", index=False)
                    batch_accum = [] # Clear after saving

        # End of repeat loop
        logging.info(f"Rank {rank} - Finished repeat {repeat+1}/{repetitions}. Total items processed in this worker so far: {total_processed_count}")

    # Save any remaining accumulated data after all repetitions
    if batch_accum:
        logging.info(f"Rank {rank} - Saving final batch accumulation. Rows: {sum(len(df) for df in batch_accum)}")
        pd.concat(batch_accum).to_parquet(tmpdir / f"multitask_predictions_{rank}_final_{uuid.uuid4()}.parquet", index=False)
        
    logging.info(f"Rank {rank} - Finished main_worker loop.")


if __name__ == "__main__":
    
    repetitions = 10
    logging.info(f"Running with {repetitions} repetitions")

    # --- Early setup for rank info ---
    rank = int(os.environ.get("RANK", -1)) # Get rank early for initial logging setup
    local_rank = int(os.environ.get("LOCAL_RANK", -1))
    world_size = int(os.environ.get("WORLD_SIZE", -1))

    # --- Setup Logging ---
    # Note: Setup function configures file logging per rank
    outdir, tmpdir = setup(rank) # Pass rank to setup

    try:
        logging.info(f"--- Starting Main Execution Block (Rank {rank}) ---")
        logging.info(f"Rank: {rank}, Local Rank: {local_rank}, World Size: {world_size}")
        dist.init_process_group("nccl", rank=rank, world_size=world_size)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
        logging.info(f"Set device to: {device}")
        logging.info(f"CUDA Device Name: {torch.cuda.get_device_name(local_rank)}")
        logging.info(f"CUDA Device Properties: {torch.cuda.get_device_properties(local_rank)}")
        logging.info(f"Initial Memory: Allocated={torch.cuda.memory_allocated(device)/1e6:.2f}MB, Reserved={torch.cuda.memory_reserved(device)/1e6:.2f}MB")

        # --- Load Model ---
        logging.info("Loading model...")
        model_load_path = "cache/train_multitask_transformer_parallel/models/moe"
        model: me.MoE = me.MoE.load(model_load_path).to(device)
        logging.info(f"Model loaded successfully from {model_load_path} and moved to {device}")

        # --- Compile/DDP ---
        logging.info("Wrapping model with DDP...")
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False) # Set find_unused_parameters=False if sure no params are unused, might speed up. Check if needed.
        model.eval()
        logging.info("Model wrapped in DDP and set to eval mode.")

        tokenizer: SelfiesPropertyValTokenizer = model.module.tokenizer # Access tokenizer from underlying module
        tokenizer.assay_indexes_tensor = torch.tensor(list(tokenizer.assay_indexes().values()), device=device)
        tokenizer.value_indexes_tensor = torch.tensor(list(tokenizer.value_indexes().values()), device=device)
        logging.info("Tokenizer setup complete.")

        # --- Configuration ---
        batch_size = 5 
        nprops = 5
        logging.info(f"Configuration: batch_size={batch_size}, nprops={nprops}, repetitions={repetitions}")

        # --- Create Context ---
        logging.info("Creating EvalContext...")
        perm_indices=list(itertools.permutations(range(nprops)))
        perm_count=math.factorial(nprops)
        context = EvalContext(
            rank=rank,
            local_rank=local_rank,
            model=model,
            tokenizer=tokenizer,
            device=device,
            perm_indices=perm_indices,
            perm_count=perm_count,
            nprops=nprops,
            batch_size=batch_size
        )
        logging.info(f"EvalContext created. Permutation count: {perm_count}")

        # --- Barrier before starting worker ---
        logging.info("Waiting at barrier before starting main worker...")
        dist.barrier()
        logging.info("Passed barrier.")

        # --- Run Main Worker ---
        logging.info(f"Starting evaluation generation on Rank {rank}")
        main_worker(context, repetitions=repetitions, outdir=outdir, tmpdir=tmpdir)
        logging.info(f"Finished main_worker call on Rank {rank}")

        # --- Barrier before finalization ---
        logging.info("Waiting at barrier before finalization...")
        dist.barrier()
        logging.info("Passed barrier.")

    except Exception as e:
        logging.exception(f"--- !!! Unhandled Exception in __main__ (Rank {rank}) !!! ---")
        # Optional: try to signal other ranks about the error? DDP might handle this partially.

    finally:
        logging.info(f"Rank {rank} entering final cleanup.")
        cleanup() # Ensure cleanup happens even on error
        logging.info(f"Rank {rank} finished cleanup.")