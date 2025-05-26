#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHONPATH=./ torchrun --standalone --nproc-per-node=8 --master-port=29500 code/5_0_0_generate_evaluations.py 2> cache/generate_evaluations/logs/err.log

import sys
import os, uuid, pathlib, shutil, logging
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
import cvae.models.datasets.restricted_dataset as rd

import cvae.utils
from cvae.tokenizer import SelfiesPropertyValTokenizer

import glob
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import hashlib
import random

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

def run_eval(i, raw_inp, raw_teach, raw_out, context: EvalContext):
    inp, teach, out = raw_inp.to(context.device), raw_teach.to(context.device), raw_out.to(context.device)

    # Filter: keep only rows with enough property-value pairs
    valid = torch.sum(torch.isin(out, context.tokenizer.value_indexes_tensor), dim=1) >= context.nprops
    inp = inp[valid]
    out = out[valid, 1:(2 * context.nprops + 1)]
    teach = teach[valid, :]

    if inp.shape[0] == 0:
        return pd.DataFrame()

    input_hashes = [hash_input(row, context.tokenizer.PAD_IDX) for row in inp]

    with torch.no_grad():
        prob = context.model(inp, teach)
        if prob is None or torch.isnan(prob).any() or torch.isinf(prob).any():
            return pd.DataFrame()
        prob = torch.softmax(prob, dim=2)  # shape: [B, T, V]

    # Define positions
    assay_pos = torch.arange(0, context.nprops * 2, 2, device=out.device)
    value_pos = assay_pos + 1

    # Extract tokens
    assays = out[:, assay_pos]             # shape: [B, nprops]
    values = out[:, value_pos]             # shape: [B, nprops]

    # Extract predictions
    prob_vals = torch.argmax(prob, dim=2)[:, value_pos]  # shape: [B, nprops]
    rawprobs = prob[:, value_pos, :][:, :, context.tokenizer.value_indexes_tensor]  # shape: [B, nprops, 2]
    probs = (rawprobs / rawprobs.sum(dim=2, keepdim=True))[:, :, 1]  # shape: [B, nprops]

    # Flatten
    assays_np       = assays.cpu().numpy().astype(str).flatten()
    values_np       = values.cpu().numpy().astype(str).flatten()
    prob_vals_np    = prob_vals.cpu().numpy().flatten()
    probs_np        = probs.cpu().numpy().flatten()
    chemical_id_np  = np.repeat(input_hashes, context.nprops)
    position        = np.tile(np.arange(context.nprops), len(input_hashes))

    # Prior context strings
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
        # shutil.rmtree(tmpdir, ignore_errors=True)
        tmpdir.mkdir(exist_ok=True, parents=True)
    
    return outdir, tmpdir

def main_worker(context: EvalContext, repetitions, outdir, tmpdir):
    logging.info(f"Rank {context.rank} - Starting main_worker.")
    logging.info(f"Rank {context.rank} - Initializing Dataset.")
    
    bp = pd.read_parquet("cache/get_benchmark_properties/benchmark_properties.parquet")
    sources = list(set(bp['source'].tolist()))

    random.seed(context.rank)
    repeat_sources = sources * repetitions
    random.shuffle(repeat_sources)
    seen_inputs = set()
    batch_accum = []
    total_processed_count = 0

    for repeat in range(len(repeat_sources)):
        
        source = repeat_sources[repeat]
        logging.info(f"Rank {context.rank} - Processing source: {source} on repeat {repeat+1} out of {len(repeat_sources)}")

        bpsource = bp[bp['source'] == source].copy()
        target_props = bpsource['property_token'].tolist()
        target_positions = [0,4]  # Positions to focus evaluation on
        sampling_weights = {prop: weight for prop, weight in zip(target_props, bpsource['weight'].tolist())}

        dataset = rd.PropertyGuaranteeDataset(
            path="cache/build_tensordataset/multitask_tensors/hld",
            tokenizer=context.tokenizer,
            nprops=context.nprops,
            target_props=target_props,
            target_positions=target_positions,
            sampling_weights=sampling_weights,
            distributed=True,
            rank=context.rank,
            world_size=dist.get_world_size()
        )

        sampler = DistributedSampler(dataset, num_replicas=dist.get_world_size(), rank=context.rank, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=context.batch_size, sampler=sampler,
            num_workers=4, pin_memory=True, prefetch_factor=5, persistent_workers=True, pin_memory_device=f"cuda:{context.local_rank}")

        logging.info(f"Rank {context.rank} - Initialized DataLoader with {len(dataloader)} batches per epoch.")
        logging.info(f"Rank {context.rank} - Using batch size: {context.batch_size}, num_workers=4")
    
        logging.info(f"Rank {context.rank} - Starting repeat {repeat+1}/{repetitions}")
        sampler.set_epoch(repeat) # Important for shuffling with DistributedSampler if shuffle=True
        
        # i, batch_data = next(enumerate(dataloader))
        for i, batch_data in tqdm.tqdm(enumerate(dataloader), total=len(dataloader), desc=f"Rank {context.rank} Repeat {repeat+1}", leave=False):

                raw_inp, raw_teach, raw_out = batch_data

                batch_tuples = tuple(map(lambda x, y: (tuple(x.tolist()), tuple(y.tolist())), raw_inp, raw_out))
                new_inputs_mask = [t not in seen_inputs for t in batch_tuples]
                seen_inputs.update(batch_tuples)

                if any(new_inputs_mask):
                    new_raw_inp = raw_inp[new_inputs_mask]
                    new_raw_teach = raw_teach[new_inputs_mask]
                    new_raw_out = raw_out[new_inputs_mask]
                    batch_df = run_eval(i, new_raw_inp, new_raw_teach, new_raw_out, context)
                    if not batch_df.empty:
                            batch_accum.append(batch_df)
                            total_processed_count += new_raw_inp.shape[0] # Count processed items

                # Check accumulation size and save periodically
                current_accum_rows = sum(len(df) for df in batch_accum)
                if batch_accum and current_accum_rows > 1_000_000: # Save threshold
                    logging.info(f"Rank {context.rank} - Saving batch accumulation at step {i}, repeat {repeat}. Accumulated rows: {current_accum_rows}")
                    pd.concat(batch_accum).to_parquet(tmpdir / f"multitask_predictions_{context.rank}_{repeat}_{i}_{uuid.uuid4()}.parquet", index=False)
                    batch_accum = [] # Clear after saving

        # End of repeat loop
        logging.info(f"Rank {context.rank} - Finished repeat {repeat+1}/{repetitions}. Total items processed in this worker so far: {total_processed_count}")

    # Save any remaining accumulated data after all repetitions
    if batch_accum:
        logging.info(f"Rank {context.rank} - Saving final batch accumulation. Rows: {sum(len(df) for df in batch_accum)}")
        pd.concat(batch_accum).to_parquet(tmpdir / f"multitask_predictions_{context.rank}_final_{uuid.uuid4()}.parquet", index=False)
        
    logging.info(f"Rank {context.rank} - Finished main_worker loop.")


if __name__ == "__main__":
    
    repetitions = 10000
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
        # model_load_path = "cache/train_multitask_transformer_parallel/models/moe"
        model_load_path = "cache/finetune_benchmarks/models/step_20000"
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
        batch_size = 1024
        nprops = 10
        logging.info(f"Configuration: batch_size={batch_size}, nprops={nprops}, repetitions={repetitions}")

        # --- Create Context ---
        logging.info("Creating EvalContext...")
        context = EvalContext(
            rank=rank,
            local_rank=local_rank,
            model=model,
            tokenizer=tokenizer,
            device=device,
            nprops=nprops,
            batch_size=batch_size
        )

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
        if dist.is_initialized():
            logging.info(f"Rank {dist.get_rank()} - Destroying process group.")
            dist.destroy_process_group()
        logging.info(f"Rank {rank} finished cleanup.")