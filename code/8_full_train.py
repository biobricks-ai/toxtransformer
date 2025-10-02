#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHONPATH=./ CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nproc-per-node=8 --master-port=29500 code/8_full_train.py 2> cache/full_train/logs/err.log

# Increase file descriptor limit
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

import shutil
import os
import pathlib
import logging
import datetime
import pickle
import hashlib

import torch
import torch.utils.data
import torch.distributed as dist
import pandas as pd

import cvae.tokenizer
import cvae.models.multitask_transformer as mt
import cvae.models.multitask_encoder as mte
import cvae.models.mixture_experts as me
from helper.trainer.selfies_properties_values_trainer import SelfiesPropertiesValuesTrainer
from helper.scheduler.warmup_cosine_trainer import WarmupCosineScheduler, WarmupCosineThenPlateau
from helper.trainer.invfreq_trainer import InverseFrequencyWeightedTrainer
from helper.trainer import AccumulatedStratifiedPropertyWeightedTrainer
from helper.trainer.evaluator import StratifiedEvaluator, StratifiedGroupEvaluator, DefaultEvaluator
from cvae.models.datasets import InMemorySelfiesPropertiesValuesDataset, BalancedSelfiesPropertiesValuesDataset, SimplePropertyMappedDataset, LazySelfiesPropertiesValuesDataset
from code.helper.utils import find_optimal_batch_size
import random
import warnings

# for compilation, shouldn't affect speed
warnings.filterwarnings("ignore", message="The CUDA Graph is empty")
torch._inductor.config.triton.cudagraphs = False

# Environment variables setup
os.environ.update({
    'MKL_THREADING_LAYER': 'GNU',
    'MKL_SERVICE_FORCE_INTEL': '0',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'PYTHONWARNINGS': 'ignore::FutureWarning',
    'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
    'NCCL_IB_DISABLE': '1',
    'NCCL_P2P_DISABLE': '1'
})


def init_logging(rank):
    """Initialize logging for rank 0 only."""
    if rank == 0:
        logdir = pathlib.Path("cache/full_train/logs")
        logdir.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            filename=logdir / "full_train.log", 
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
            filemode='w'
        )
        logging.info("Logging initialized.")

def create_model_and_optimizer(tokenizer, base_lr):
    """Create model and optimizer."""
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # OLDRUN
    # 2025-09-07 22:27:11,274 - INFO - root - ✅ EVAL COMPLETE [Step 15550] - Loss: 0.6115 (best: 0.5151), AUC: 0.8980 (best: 0.8999), BAC: 0.8085, LR: 3.95e-04
    # config = mte.MultitaskEncoderConfig(
    #     hdim=128, 
    #     nhead=8,
    #     ff_mult=4,
    #     num_layers=16,
    #     activation='gelu',
    #     dropout_rate=0.3, 
    #     attention_dropout=0.0,
    #     layer_dropout=0.0,
    #     output_size=2
    # )

    # 2025-09-07 23:53:28,815 - INFO - root - ✅ EVAL COMPLETE [Step 7010] - Loss: 0.5753 (best: 0.5367), AUC: 0.9077 (best: 0.9077), BAC: 0.8144, LR: 7.04e-04 🎯 BEST AUC!
    config = mte.MultitaskEncoderConfig(
        hdim=128,
        nhead=8,
        ff_mult=4,
        num_layers=16,
        activation='gelu',
        dropout_rate=0.3, 
        attention_dropout=0.1,
        layer_dropout=0.1,
        output_size=2
    )


    model = mte.MultitaskEncoder(tokenizer=tokenizer, config=config)
    # model = mte.MultitaskEncoder.load('cache/full_train/models/mt/best_loss')
    model = torch.compile(model, mode='default', fullgraph=True, dynamic=False)

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=base_lr,
        betas=(0.9, 0.95),
        weight_decay=.1,
        eps=1e-8
    )
    
    return model, optimizer

def create_dataloaders(tokenizer, trnds, valds, batch_size, world_size, rank):
    """Create training and validation dataloaders."""
    # Reduced settings to prevent memory leaks
    cpus_per_rank = max(2, os.cpu_count() // world_size)
    train_workers = max(2, min(4, cpus_per_rank))  # Reduced
    val_workers = max(1, min(2, cpus_per_rank))    # Reduced
    prefetch_factor = 10  # Significantly reduced from 40

    # Create dataloaders
    trndl = torch.utils.data.DataLoader(
        trnds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=train_workers,
        pin_memory=True,          # Keep this for speed
        persistent_workers=False,  # Keep this for speed  
        prefetch_factor=prefetch_factor,  # Much lower
        sampler=torch.utils.data.distributed.DistributedSampler(
            trnds, num_replicas=world_size, rank=rank, drop_last=True, shuffle=True
        )
    )

    valdl = torch.utils.data.DataLoader(
        valds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=False,
        prefetch_factor=prefetch_factor,
        sampler=torch.utils.data.distributed.DistributedSampler(
            valds, num_replicas=world_size, rank=rank, drop_last=True, shuffle=True
        )
    )
    
    return trndl, valdl

def log_training_info(rank, model, trnds, valdl, trndl, params, batch_size, world_size):
    """Log training information."""
    if rank != 0:
        return
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"=== Training Configuration ===")
    logging.info(f"Model parameters: {num_params/1e6:.2f} million")
    logging.info(f"Training samples: {len(trnds)}")
    logging.info(f"Batch size: {batch_size}, World size: {world_size}")
    logging.info(f"Effective accumulation batch size: {params['effective_accum_batch_size']}")
    logging.info(f"Training epochs: {params['training_epochs']}")
    logging.info(f"Training max steps: {params['training_max_steps']}")
    logging.info(f"Batches in epoch: {params['batches_in_epoch']}")
    logging.info(f"Scheduler warmup steps: {params['scheduler_warmup_accum_steps']}")
    logging.info(f"Train batches: {len(trndl)}")
    logging.info(f"Validation batches: {len(valdl)}")
    logging.info(f"Warmup as % of total: {(params['scheduler_warmup_accum_steps'] / params['training_max_steps']) * 100:.2f}%")

def calculate_training_params(trnds, valds, batch_size, world_size, all_properties):
    """Calculate training parameters."""
    # Training configuration
    training_epochs = 100
    # target_effective_batch_size = batch_size * world_size * 8 # 8 gradient accumulation steps OLD RUN
    target_effective_batch_size = batch_size * world_size * 2

    # Calculate steps and accumulation
    batches_in_epoch = max(len(trnds) // (batch_size * world_size), 1)
    training_max_steps = training_epochs * batches_in_epoch
    num_batches_to_accumulate = max(target_effective_batch_size // (batch_size * world_size), 1)
    effective_accum_batch_size = batch_size * world_size * num_batches_to_accumulate
    
    # validation params
    desired_validation_samples = 100_000
    validation_batches = max(desired_validation_samples // (batch_size * world_size), 1)

    # Scheduler parameters
    accum_steps_in_epoch = max(batches_in_epoch // num_batches_to_accumulate, 1)
    # scheduler_warmup_accum_steps = max(100, 4*int(accum_steps_in_epoch)) # warm up over 4 epochs
    scheduler_warmup_accum_steps = 5000
    # scheduler_accum_steps_in_first_cycle = scheduler_warmup_accum_steps * 30 # cosine for 30 epochs
    scheduler_accum_steps_in_first_cycle = 25_000
    
    return {
        'training_max_steps': training_max_steps,
        'effective_accum_batch_size': effective_accum_batch_size,
        'scheduler_warmup_accum_steps': scheduler_warmup_accum_steps,
        'scheduler_accum_steps_in_first_cycle': scheduler_accum_steps_in_first_cycle,
        'batches_in_epoch': batches_in_epoch,
        'accum_steps_in_epoch': accum_steps_in_epoch,
        'validation_batches': validation_batches,
        'training_epochs': training_epochs
    }

def main(rank, world_size):
    """Main training function."""
    init_logging(rank)
    
    # Setup directories
    outdir = pathlib.Path("cache/full_train")
    cachedir = outdir / "cache"
    cachedir.mkdir(parents=True, exist_ok=True)
    modeldir = outdir / "models"
    metricsdir = outdir / "metrics"
    
    logging.info(f"Rank {rank} starting setup.")
    
    # Load tokenizer
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
    
    # Dataset configuration
    all_properties = list(range(tokenizer.num_assays))
    trnpaths = list(pathlib.Path("cache/build_tensordataset/multitask_tensors/trn").glob("*.pt"))
    tstpaths = list(pathlib.Path("cache/build_tensordataset/multitask_tensors/hld").glob("*.pt"))
    allpaths = trnpaths + tstpaths

    trnds = SimplePropertyMappedDataset(
        paths=allpaths,
        tokenizer=tokenizer,
        target_properties=all_properties,
        nprops=5
    )
    logging.info(f"Training dataset loaded: {len(trnds)} samples")

    logging.info("Loading validation dataset (checking Redis cache first)...")
    valds = SimplePropertyMappedDataset(
        paths=tstpaths,
        tokenizer=tokenizer,
        target_properties=all_properties,
        nprops=1,
    )
    logging.info(f"Validation dataset loaded: {len(valds)} samples")

    logging.info("Getting optimal batch size...")

    min_lr = 1e-4
    max_lr = 1e-3
    model, optimizer = create_model_and_optimizer(tokenizer, base_lr=min_lr)

    batch_size = 4800
    
    # Create dataloaders
    logging.info("Creating dataloaders...")
    trndl, valdl = create_dataloaders(tokenizer, trnds, valds, batch_size, world_size, rank)

    # Calculate training parameters
    logging.info("Calculating training params...")
    params = calculate_training_params(trnds, valds, batch_size, world_size, all_properties)

    scheduler = WarmupCosineThenPlateau(
        optimizer,
        warmup_steps=params["scheduler_warmup_accum_steps"],
        cosine_cycle_length=params["scheduler_accum_steps_in_first_cycle"],
        min_lr=min_lr,
        max_lr=max_lr,
        plateau_mode="min",
        plateau_factor=0.5,
        plateau_patience=100,
        plateau_min_lr=1e-5,
        plateau_verbose=True,
    )
    
    # Log training info
    log_training_info(rank, model, trnds, valdl, trndl, params, batch_size, world_size)

    # Setup evaluator and trainer
    stratified_eval = StratifiedEvaluator(num_tasks=len(tokenizer.assay_indexes()), rank=rank)
    # stratified_eval = StratifiedGroupEvaluator(rank=rank, num_tasks=len(tokenizer.assay_indexes()))
    
    desired_training_samples_before_eval = 5_000_000
    eval_every = desired_training_samples_before_eval // (batch_size * world_size)

    trainer = AccumulatedStratifiedPropertyWeightedTrainer(
        model=model,
        rank=rank,
        tokenizer=tokenizer,
        trn_iterator=trndl,
        batch_size=batch_size,
        scheduler=scheduler,
        max_steps=params['training_max_steps'],
        effective_accum_batch_size=params['effective_accum_batch_size'],
        eval_samples=params["validation_batches"],
        first_eval=5,
        eval_every=eval_every,
        find_unused_parameters=False,
        evaluator=stratified_eval,
        # ema_alpha=.2, OLDRUN
        # max_score=10.0, OLDRUN
        ema_alpha=.5,
        max_score=1.0,
        min_score=1.0,
        skip_initial_batches=5
    )

    # Setup trainer
    trainer.set_validation_dataloader(valdl)
    trainer.set_model_savepath(modeldir / "mt")
    trainer.set_metrics_file(metricsdir / "multitask_loss.tsv", overwrite=True)
    
    logging.info(f"Rank {rank} starting training.")
    trainer.start()

if __name__ == "__main__":
    # Get distributed training parameters
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Setup distributed training
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=30)
    )
    
    print(f"RANK {rank} starting with WORLD_SIZE {world_size}")
    
    try:
        main(rank, world_size)
    finally:
        # Cleanup
        dist.barrier()
        dist.destroy_process_group()

    # copy best model to brick location
    shutil.copytree(
        "cache/full_train/models/mt/best_loss", 
        "brick/multitask_transformer_model", 
        dirs_exist_ok=True
    )