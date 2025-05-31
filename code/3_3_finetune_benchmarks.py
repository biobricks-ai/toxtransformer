#!/usr/bin/env python
# -*- coding: utf-8 -*-
# CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nproc-per-node=8 --master-port=29500 code/3_3_finetune_benchmarks.py 2> cache/finetune_benchmarks/logs/err.log

import os
import sys
import shutil
import pathlib
import logging
import datetime
from typing import Tuple

import torch
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist

import cvae.tokenizer
import cvae.utils
import cvae.models.multitask_transformer as mt
import cvae.models.datasets.restricted_dataset as rd
import pandas as pd
import cvae.models.mixture_experts as me
from helper.trainer.trainer import Trainer
from helper.trainer.restricted_trainer import RestrictedTrainer, create_restricted_trainer

# Environment variables setup
os.environ['MKL_THREADING_LAYER'] = 'GNU'
os.environ['MKL_SERVICE_FORCE_INTEL'] = '0'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['PYTHONWARNINGS'] = 'ignore::FutureWarning'
# os.environ['NCCL_DEBUG'] = 'INFO'
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
# os.environ['TORCH_NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_IB_DISABLE'] = '0'
os.environ['NCCL_P2P_DISABLE'] = '0'

# Enable TF32 for A100s
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
mp.set_start_method("fork", force=True)

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30))

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def init_logging(rank):
    logdir = pathlib.Path("cache/finetune_benchmarks/logs")
    logdir.mkdir(exist_ok=True, parents=True)
    logging.basicConfig(filename=logdir / f"finetune_benchmarks_rank{rank}.log", level=logging.INFO, 
                        format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
    logging.info(f"Rank {rank}: Logging initialized.")

def main(rank, world_size):
    setup(rank, world_size)
    init_logging(rank)

    outdir = pathlib.Path("cache/finetune_benchmarks")
    modeldir = outdir / "models"
    metricsdir = outdir / "metrics"
    
    # Create directories if they don't exist
    modeldir.mkdir(exist_ok=True, parents=True)
    metricsdir.mkdir(exist_ok=True, parents=True)

    logging.info(f"Rank {rank} starting setup.")

    # Load from training step
    model = me.MoE.load("cache/train_multitask_transformer_parallel/models/moe")
    tokenizer = model.tokenizer
    batch_size = 240
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    cpus_per_rank = max(2, (os.cpu_count() or 8) // world_size)
    train_workers = max(2, min(4, cpus_per_rank))
    val_workers = max(1, min(2, cpus_per_rank))
    
    bp = pd.read_parquet("cache/get_benchmark_properties/benchmark_properties.parquet")
    target_props = bp['property_token'].tolist()
    target_positions = [0, 4]  # Positions to focus evaluation on
    sampling_weights = {prop: weight for prop, weight in zip(target_props, bp['weight'].tolist())}
    nprops = 5

    trnds = rd.PropertyGuaranteeDataset(
        path="cache/build_tensordataset/multitask_tensors/trn",
        tokenizer=tokenizer,
        nprops=nprops,
        target_props=target_props,
        target_positions=target_positions,
        sampling_weights=sampling_weights,
        distributed=True,
        rank=rank,
        world_size=world_size
    )

    valds = rd.PropertyGuaranteeDataset(
        path="cache/build_tensordataset/multitask_tensors/tst",
        tokenizer=tokenizer,
        nprops=nprops,
        target_props=target_props,
        target_positions=target_positions,
        sampling_weights=sampling_weights,
        distributed=True,
        rank=rank,
        world_size=world_size
    )

    trndl = torch.utils.data.DataLoader(
        trnds, batch_size=batch_size, shuffle=False, num_workers=train_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=20,
        sampler=torch.utils.data.distributed.DistributedSampler(trnds, num_replicas=world_size, rank=rank, drop_last=True)
    )

    valdl = torch.utils.data.DataLoader(
        valds, batch_size=batch_size, shuffle=False, num_workers=val_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=20,
        sampler=torch.utils.data.distributed.DistributedSampler(valds, num_replicas=world_size, rank=rank, drop_last=True)
    )

    # Create a RestrictedTrainer using the updated factory function
    trainer = create_restricted_trainer(
        model=model,
        rank=rank,
        tokenizer=tokenizer,
        train_iterator=trndl,
        batch_size=batch_size,
        target_properties=target_props,
        target_positions=target_positions,
        nprops=nprops,
        scheduler_min_lr=1e-4,
        scheduler_max_lr=2e-4,
        scheduler_warmup_steps=500,
        scheduler_max_steps=3e5,
        max_steps=1e7,
        savepath=modeldir / "moe",
        validation_dataloader=valdl,
        first_eval=5,
        eval_every=10000,
        eval_samples=400
    )
    trainer.set_metrics_file(metricsdir / "finetune_benchmarks_loss.tsv", overwrite=True)
    trainer.set_mask_percent(0.1)
    
    # Set evaluation positions to focus on positions 0 and 4
    trainer.set_evaluation_positions(eval_positions=[0, 4])
    trainer.set_mask_percent(0.1)

    if rank == 0:
        logging.info(f"trnds samples: {len(trnds)}, valds samples: {len(valds)}")
        logging.info(f"workers train: {train_workers}, val: {val_workers}")
        trainer.log(f"{len(trndl)} train batches")
        trainer.log(f"{len(valdl)} validation batches")
        trainer.log(f"{num_params/1e6:.2f} million parameters")
        trainer.log(f"Gradient accumulation: {trainer.gradient_accumulation_steps}")
        trainer.log(f"Evaluating on restricted positions: [0, 4]")

    trainer.start()
    cleanup()

    if rank == 0:
        logging.info("Copying final model from modeldir to brick/moe...")
        # save modeldir/moe to brick/moe
        # delete brick/moe directory
        if os.path.exists("brick/moe"):
            shutil.rmtree("brick/moe", ignore_errors=True)
        shutil.copytree(modeldir / "moe", "brick/moe")

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"RANK {rank} starting with WORLD_SIZE {world_size}")
    main(rank, world_size)