#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHONPATH=./ CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nproc-per-node=8 --master-port=29500 code/3_1_1_train_multitask_transformer_parallel.py 2> cache/train_multitask_transformer_parallel/logs/err.log

import os
import sys
import shutil
import pathlib
import logging
import datetime

import torch
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import pandas as pd

import cvae.tokenizer
import cvae.utils
import cvae.models.multitask_transformer as mt
# import cvae.models.datasets.sampling_dataset as sd
import cvae.models.mixture_experts as me
from helper.trainer.trainer import Trainer
from cvae.models.datasets.inmemory_sequence_shift_dataset import InMemorySequenceShiftDataset
import random

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
# torch.backends.cudnn.benchmark = True
# torch.backends.cuda.matmul.allow_tf32 = True
# torch.backends.cudnn.allow_tf32 = True
# mp.set_start_method("fork", force=True)

def init_logging(rank):
    if rank == 0:
        logdir = pathlib.Path("cache/train_multitask_transformer_parallel/logs")
        logdir.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(filename=logdir / "train_multitask_transformer_parallel.log", level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
        logging.info("Logging initialized.")

def main(rank, world_size):
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    init_logging(rank)

    outdir = pathlib.Path("cache/train_multitask_transformer_parallel")
    modeldir = outdir / "models"
    metricsdir = outdir / "metrics"

    logging.info(f"Rank {rank} starting setup.")
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

    model = me.MoE(
        tokenizer, 
        num_experts=32,  # Reduced from 32 for speed
        k=2,           # Reduced from 4 for efficiency  
        hdim=256 // 2,      # Reduced from 256
        dim_feedforward=1024 // 2,  # Reduced from 1024
        nhead=2, 
        expert_layers=6 // 2,  # Reduced from 6
        balance_loss_weight=0.0, 
        diversity_loss_weight=0,
        noise_factor=0.1,           # Initial noise
        noise_decay_steps=1000,    # Steps to decay over
        dropout_rate=.1
    )
    model = me.MoE.load("cache/train_multitask_transformer_parallel/models/pretrained") #result of training with no frozen params or boosters
    batch_size = 32 * 3

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    first_expert = model.experts[0]
    params_in_expert = sum(p.numel() for p in first_expert.parameters() if p.requires_grad)

    logging.info(f"Rank {rank} model parameters: {num_params/1e6:.2f} million, params in first expert: {params_in_expert/1e6:.2f} million")
    cpus_per_rank = max(2, os.cpu_count() // world_size)
    train_workers = max(2, min(32, cpus_per_rank))
    val_workers = max(1, min(20, cpus_per_rank))

    bp = pd.read_parquet("cache/get_benchmark_properties/benchmark_properties.parquet")
    benchmark_properties = list(bp['property_token'].unique())
    tox21_props = list(bp[bp['source'].isin(['Tox21'])]['property_token'].unique().tolist())

    # tmp = InMemorySequenceShiftDataset("cache/build_tensordataset/multitask_tensors/tmp", tokenizer, nprops=10)
    # trnds = tmp
    # valds = tmp

    # 90% of samples have less than 42 props, so nprops=42 is a good choice
    trnds = InMemorySequenceShiftDataset("cache/build_tensordataset/multitask_tensors/trn", tokenizer, nprops=42)
    valds = InMemorySequenceShiftDataset("cache/build_tensordataset/multitask_tensors/tst", tokenizer, nprops=42, assay_filter=tox21_props)

    trndl = torch.utils.data.DataLoader(
        trnds, batch_size=batch_size, shuffle=False, num_workers=train_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=100,
        sampler=torch.utils.data.distributed.DistributedSampler(trnds, num_replicas=world_size, rank=rank, drop_last=True, shuffle=True)
    )

    valdl = torch.utils.data.DataLoader(
        valds, batch_size=batch_size, shuffle=False, num_workers=val_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=100,
        sampler=torch.utils.data.distributed.DistributedSampler(valds, num_replicas=world_size, rank=rank, drop_last=True, shuffle=True)
    )

    num_batches_to_accumulate = 8000 // (batch_size * world_size) # keeping our effective batch size at 8000
    effective_accum_batch_size = batch_size * world_size * num_batches_to_accumulate
    
    # - warmup for half the first epoch 
    # - cycle with increasing length starting at .5 epoch
    # - train for 3 epochs
    # - an accum step occurs once every `num_batches_to_accumulate` batches
    # - the scheduler steps once every accumulation step
    batches_in_epoch = len(trnds) // (batch_size * world_size)
    training_epochs = 10
    training_max_steps = training_epochs * batches_in_epoch

    accum_steps_in_epoch = batches_in_epoch // num_batches_to_accumulate
    scheduler_warmup_accum_steps = int(accum_steps_in_epoch * training_epochs * .05)
    scheduler_warmup_accum_steps = max(scheduler_warmup_accum_steps, 100)  
    scheduler_accum_steps_in_first_cycle = accum_steps_in_epoch * 3

    if rank == 0:
        logging.info(f"Model parameters: {num_params/1e6:.2f} million")
        logging.info(f"trnds samples: {len(trnds)}, valds samples: {len(valds)}")
        logging.info(f"Batch size: {batch_size}, World size: {world_size}")
        logging.info(f"Batches to accumulate: {num_batches_to_accumulate}")
        logging.info(f"Effective accumulation batch size: {effective_accum_batch_size}")
        logging.info(f"Batches in epoch: {batches_in_epoch}")
        logging.info(f"Accumulation steps in epoch: {accum_steps_in_epoch}")
        logging.info(f"Scheduler warmup accumulation steps: {scheduler_warmup_accum_steps}")
        logging.info(f"Scheduler accumulation steps in first cycle: {scheduler_accum_steps_in_first_cycle}")
        logging.info(f"Training epochs: {training_epochs}")
        logging.info(f"Training max steps: {training_max_steps}")
        logging.info(f"Warmup steps as percentage of total: { (scheduler_warmup_accum_steps * num_batches_to_accumulate) / (training_max_steps) * 100:.2f}%")

    trainer = Trainer(model, rank, tokenizer, trndl, batch_size=batch_size, 
        scheduler_warmup_steps=scheduler_warmup_accum_steps, 
        scheduler_max_steps=scheduler_accum_steps_in_first_cycle, 
        scheduler_min_lr=1e-6, 
        scheduler_max_lr=1e-3,   
        effective_accum_batch_size=effective_accum_batch_size,
        max_steps=training_max_steps, eval_samples=2_000, first_eval=5, eval_every=1000)
    
    trainer.set_validation_dataloader(valdl)
    trainer.set_mask_percent(0.1)
    trainer.set_model_savepath(modeldir / "moe")
    trainer.set_metrics_file(metricsdir / "multitask_loss.tsv", overwrite=True)

    if rank == 0:
        logging.info(f"Training epochs: {training_epochs}, Accumulation steps in epoch: {accum_steps_in_epoch}")
        logging.info(f"trnds samples: {len(trnds)}, valds samples: {len(valds)}")
        logging.info(f"workers train: {train_workers}, val: {val_workers}")
        logging.info(f"{len(trndl)} train batches")
        logging.info(f"{len(valdl)} validation batches")
        logging.info(f"{num_params/1e6:.2f} million parameters")
        logging.info(f"Gradient accumulation: {trainer.gradient_accumulation_steps}")

    trainer.start()

def train_frozen_shared_embeddings(trainer):
    """Train with frozen shared embeddings"""
    if trainer.rank == 0:
        logging.info("Starting training with frozen shared embeddings...")
    
    # Freeze shared embeddings
    trainer.model.freeze_shared_embeddings()
    
    # Update trainer parameters for frozen embeddings training
    trainer.scheduler_warmup_steps = trainer.scheduler_warmup_steps // 2  # Reduce warmup
    trainer.scheduler_max_lr = 1e-6  # Lower learning rate for frozen training
    trainer.scheduler_min_lr = 3e-4  # Lower minimum learning rate
    trainer.max_steps = trainer.max_steps // 2  # Train for fewer steps
    
    # Set new model save path
    trainer.set_model_savepath(trainer.model_savepath.parent / "moe_frozen_embeddings")
    trainer.set_metrics_file(trainer.metrics_file.parent / "multitask_loss_frozen_embeddings.tsv", overwrite=True)
    
    if trainer.rank == 0:
        frozen_params = sum(1 for p in trainer.model.parameters() if not p.requires_grad)
        total_params = sum(1 for p in trainer.model.parameters())
        logging.info(f"Frozen parameters: {frozen_params}/{total_params}")
    
    trainer.start()
    
if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=30),
    )
    
    print(f"RANK {rank} starting with WORLD_SIZE {world_size}")
    trainer = main(rank, world_size)
    dist.barrier()  # Ensure all processes finish before cleanup
    train_frozen_shared_embeddings(trainer)

    dist.destroy_process_group()