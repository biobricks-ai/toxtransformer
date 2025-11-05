#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
FOLD=0 PYTHONPATH=./ CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nproc-per-node=8 \
--master-port=29500 code/3_1_1_train_multitask_transformer_parallel.py 2> \
cache/train_multitask_transformer_parallel/logs/err.log
"""
# Increase file descriptor limit
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

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
import cvae.models.datasets.custom_sampler as custom_sampler

from helper.scheduler.warmup_cosine_trainer import WarmupCosineScheduler, WarmupCosineThenPlateau
from helper.trainer.invfreq_trainer import InverseFrequencyWeightedTrainer

from helper.trainer import AccumulatedStratifiedPropertyWeightedTrainer
from helper.trainer.evaluator import StratifiedEvaluator, StratifiedGroupEvaluator, DefaultEvaluator
from cvae.models.datasets import InMemorySelfiesPropertiesValuesDataset, SimplePropertyMappedDataset
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
})


def init_logging(rank):
    """Initialize logging for rank 0 only."""
    if rank == 0:
        logdir = pathlib.Path("cache/train_multitask_transformer_parallel/logs")
        logdir.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            filename=logdir / "train_multitask_transformer_parallel.log", 
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

    model = mte.create_multitask_encoder(model_size='V3', tokenizer=tokenizer)
    # model = mte.MultitaskEncoder.load("cache/train_multitask_transformer_parallel/models/me_roundrobin_property_dropout/best_loss")
    model = torch.compile(model, mode='default', fullgraph=True, dynamic=False)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr, betas=(0.9, 0.98), weight_decay=1e-2, eps=1e-8, fused=False  
    )
    
    return model, optimizer

def create_testing_datasets(tokenizer):
    tmppaths = list(pathlib.Path("cache/build_tensordataset/multitask_tensors/trn").glob("*.pt"))
    trnds = InMemorySelfiesPropertiesValuesDataset(tmppaths[:30], tokenizer, nprops=5)
    valds = InMemorySelfiesPropertiesValuesDataset(tmppaths[:20], tokenizer, nprops=1)
    return trnds, valds
    
def create_datasets(tokenizer, rank, nprops=20):
    split = os.getenv('SPLIT')
    trnpaths = list(pathlib.Path(f"cache/build_tensordataset/bootstrap/split_{split}/train").glob("*.pt"))
    tstpaths = list(pathlib.Path(f"cache/build_tensordataset/bootstrap/split_{split}/test").glob("*.pt"))
    trnds = SimplePropertyMappedDataset(
        paths=trnpaths,
        tokenizer=tokenizer,
        target_properties=list(range(tokenizer.num_assays)),
        nprops=nprops,
        seed=rank
    )
    valds = InMemorySelfiesPropertiesValuesDataset(tokenizer=tokenizer, paths=tstpaths, nprops=1)
    return trnds, valds

def create_dataloaders(tokenizer, trnds, valds, batch_size, world_size, rank):
    """Create training and validation dataloaders."""
    
    cpus_per_rank = max(2, os.cpu_count() // world_size)
    train_workers = max(2, min(12, cpus_per_rank))
    val_workers = max(1, min(8, cpus_per_rank))
    prefetch_factor = 20

    trndl = torch.utils.data.DataLoader(
        trnds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=train_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
        # sampler=custom_sampler.FastDistributedSampler(
        #     dataset=trnds,
        #     num_replicas=world_size,
        #     rank=rank,
        #     shuffle=True,
        #     seed=42,
        #     drop_last=True
        # )
    )

    valdl = torch.utils.data.DataLoader(
        valds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=val_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=prefetch_factor,
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
    logging.info(f"Training max steps: {params['training_max_steps']}")
    logging.info(f"Scheduler warmup steps: {params['scheduler_warmup_accum_steps']}")
    logging.info(f"Train batches: {len(trndl)}")
    logging.info(f"Validation batches: {len(valdl)}")
    logging.info(f"Warmup as % of total: {(params['scheduler_warmup_accum_steps'] / params['training_max_steps']) * 100:.2f}%")

def calculate_training_params(trnds, valds, batch_size, world_size, all_properties):
    """Calculate training parameters."""

    # Calculate steps and accumulation
    target_effective_batch_size = batch_size * world_size * 6
    num_batches_to_accumulate = max(target_effective_batch_size // (batch_size * world_size), 1)
    effective_accum_batch_size = batch_size * world_size * num_batches_to_accumulate
    
    # validation params
    desired_validation_samples = len(all_properties) * 2 * 100 # 100 samples per property, positive and negative
    validation_batches = max(desired_validation_samples // (batch_size * world_size), 1)

    # Scheduler parameters
    scheduler_warmup_accum_steps = 1000
    scheduler_accum_steps_in_first_cycle = 30_000
    
    return {
        'training_max_steps': 200_000 * 6,
        'effective_accum_batch_size': effective_accum_batch_size,
        'scheduler_warmup_accum_steps': scheduler_warmup_accum_steps,
        'scheduler_accum_steps_in_first_cycle': scheduler_accum_steps_in_first_cycle,
        'validation_batches': validation_batches,
    }

def main(rank, world_size):
    """Main training function."""
    init_logging(rank)
    logging.info(f"Rank {rank} starting setup.")

    # Setup directories
    outdir = pathlib.Path("cache/train_multitask_transformer_parallel")
    modeldir = outdir / "models"
    metricsdir = outdir / "metrics"

    # Load tokenizer
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
    # trnds, valds = create_testing_datasets(tokenizer)
    trnds, valds = create_datasets(tokenizer, rank)

    # Create dataloaders
    logging.info("Creating dataloaders...")
    batch_size = 1000
    trndl, valdl = create_dataloaders(tokenizer, trnds, valds, batch_size, world_size, rank)

    # Calculate training parameters
    logging.info("Calculating training params...")
    all_properties = list(range(tokenizer.num_assays))
    params = calculate_training_params(trnds, valds, batch_size, world_size, all_properties)

    # Create model, optimizer and scheduler
    min_lr = 1e-6
    model, optimizer = create_model_and_optimizer(tokenizer, base_lr=min_lr)
    scheduler = WarmupCosineThenPlateau(
        optimizer,
        warmup_steps=params["scheduler_warmup_accum_steps"],
        cosine_cycle_length=params["scheduler_accum_steps_in_first_cycle"],
        warmup_start_lr=min_lr,
        warmup_end_lr=1e-4,
        cosine_start_lr=1e-4,  # starts at max_lr (same as warmup_end_lr)
        cosine_end_lr=5e-5,    # ends at min_lr
        plateau_start_lr=5e-5, # plateau starts at min_lr (same as cosine_end_lr)
        plateau_end_lr=1e-6,     # was plateau_min_lr
        plateau_mode="min",
        plateau_factor=0.5,
        plateau_patience=500, # minimum of n=log2​(100)≈6.64 halvings and 6 * 5000 = 30k steps
        plateau_verbose=True,
    )
    dist.barrier()

    # Log training info
    log_training_info(rank, model, trnds, valdl, trndl, params, batch_size, world_size)

    # Setup evaluator and trainer
    label_smoothing = 0.15
    stratified_eval = StratifiedGroupEvaluator(tokenizer=tokenizer, rank=rank, label_smoothing=label_smoothing)
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
        first_eval=0,
        eval_every=1250,
        find_unused_parameters=False,
        evaluator=stratified_eval,
        
        # No accumulation needed
        acc_coalesce_every=1,
        
        # Moderate weighting
        max_score=1.0,
        min_score=1.0,
        
        # Balanced sampling
        property_sample_rate=1.0,
        sample_bias_strength=0.0,
        
        # Faster EMA due to large batch
        ema_alpha=0.9,
        
        # Adjust for large batch size
        skip_initial_batches=10,
        sampling_warmup=20,

        label_smoothing=label_smoothing,
        unsampled_property_weight=0.0
    )

    # Setup trainer
    trainer.set_validation_dataloader(valdl)
    trainer.set_model_savepath(modeldir / "me_roundrobin_property_dropout_V3")
    trainer.set_metrics_file(metricsdir / "multitask_loss.tsv", overwrite=True)
    
    logging.info(f"Rank {rank} starting training.")
    trainer.start()

if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", "0"))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

    # Bind this process to its local GPU
    torch.cuda.set_device(RANK)

    # Initialize the process group (using Gloo to avoid NCCL segfaults)
    # TODO don't use gloo once off of lambdalabs
    dist.init_process_group(
        backend="gloo",  # Changed from nccl due to segfaults
        init_method="env://",
        rank=RANK,
        world_size=WORLD_SIZE,
        timeout=datetime.timedelta(minutes=30),
    )

    print(f"[Init] RANK={RANK} WORLD_SIZE={WORLD_SIZE} GPU={RANK}")
    main(rank=RANK, world_size=WORLD_SIZE)
    