#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHONPATH=./ CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nproc-per-node=8 --master-port=29500 code/3_1_2_train_adapters_all_props.py 2> cache/train_adapters_all_props/logs/err.log

# Increase file descriptor limit
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

import os
import pathlib
import logging
import datetime

import torch
import torch.utils.data
import torch.distributed as dist
import pandas as pd

import cvae.tokenizer
import cvae.models.adapter_model as adapter_model
from helper.trainer.selfies_properties_values_trainer import SelfiesPropertiesValuesTrainer
from helper.scheduler.warmup_cosine_trainer import WarmupCosineScheduler, WarmupCosineThenPlateau
from helper.trainer.invfreq_trainer import InverseFrequencyWeightedTrainer
from helper.trainer import AccumulatedStratifiedPropertyWeightedTrainer
from helper.trainer.evaluator import StratifiedGroupEvaluator

from cvae.models.datasets import InMemorySelfiesPropertiesValuesDataset, SimplePropertyMappedDataset
from code.helper.utils import find_optimal_batch_size
import random

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
        logdir = pathlib.Path("cache/train_adapters_all_props/logs")
        logdir.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(
            filename=logdir / "train_adapters_all_props.log", 
            level=logging.INFO, 
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s', 
            filemode='w'
        )
        logging.info("Logging initialized.")

def create_model_and_optimizer(tokenizer, benchmark_properties, shared_base_path, base_lr):
    """Create adapter model and optimizer."""
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # Create adapter model with all benchmark properties
    model = adapter_model.AdapterTransformer(
        tokenizer=tokenizer,
        shared_base_path=shared_base_path,
        benchmark_properties=benchmark_properties,
        adapter_hidden_dim=24,  # ~8.8M params total with 4k properties
        adapter_dropout=0.1
    )

    # Only optimize the adapter parameters (shared base is frozen)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=base_lr, 
        betas=(0.9, 0.99), 
        weight_decay=1e-2
    )
    
    return model, optimizer

def create_testing_datasets(tokenizer):
    tmppaths = list(pathlib.Path("cache/build_tensordataset/multitask_tensors/trn").glob("*.pt"))
    trnds = InMemorySelfiesPropertiesValuesDataset(tmppaths[:30], tokenizer, nprops=5)
    valds = InMemorySelfiesPropertiesValuesDataset(tmppaths[:10], tokenizer, nprops=1)
    return trnds, valds
    
def create_datasets(tokenizer, rank):
    trnpaths = list(pathlib.Path("cache/build_tensordataset/multitask_tensors/trn").glob("*.pt"))
    tstpaths = list(pathlib.Path("cache/build_tensordataset/multitask_tensors/hld").glob("*.pt"))
    trnds = SimplePropertyMappedDataset(
        paths=trnpaths,
        tokenizer=tokenizer,
        target_properties=list(range(tokenizer.num_assays)),
        nprops=5,
        seed=rank
    )
    valds = InMemorySelfiesPropertiesValuesDataset(tokenizer=tokenizer, paths=tstpaths, nprops=1)
    return trnds, valds

def create_dataloaders(tokenizer, trnds, valds, batch_size, world_size, rank):
    """Create training and validation dataloaders."""
    # Calculate workers
    cpus_per_rank = max(2, os.cpu_count() // world_size)
    train_workers = max(2, min(4, cpus_per_rank))
    val_workers = max(1, min(2, cpus_per_rank))
    prefetch_factor = 20

    # Create dataloaders
    trndl = torch.utils.data.DataLoader(
        trnds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=train_workers,
        pin_memory=False, 
        persistent_workers=False, 
        prefetch_factor=prefetch_factor,
        sampler=torch.utils.data.distributed.DistributedSampler(
            trnds, num_replicas=world_size, rank=rank, drop_last=True, shuffle=True
        )
    )

    valdl = torch.utils.data.DataLoader(
        valds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=val_workers,
        pin_memory=False, 
        persistent_workers=False, 
        prefetch_factor=prefetch_factor,
        sampler=torch.utils.data.distributed.DistributedSampler(
            valds, num_replicas=world_size, rank=rank, drop_last=True, shuffle=True
        )
    )
    
    return trndl, valdl

def calculate_training_params(trnds, valds, batch_size, world_size):
    """Calculate training parameters."""
    # Training configuration - shorter training for adapters
    training_epochs = 10  # Adapters need less training
    target_effective_batch_size = int(batch_size * world_size * 4)  # Smaller accumulation

    # Calculate steps and accumulation
    batches_in_epoch = max(len(trnds) // (batch_size * world_size), 1)
    training_max_steps = training_epochs * batches_in_epoch
    num_batches_to_accumulate = max(target_effective_batch_size // (batch_size * world_size), 1)
    effective_accum_batch_size = batch_size * world_size * num_batches_to_accumulate
    
    # validation params
    desired_validation_samples = 50_000
    validation_batches = max(desired_validation_samples // (batch_size * world_size), 1)

    # Scheduler parameters - faster warmup for adapters
    accum_steps_in_epoch = max(batches_in_epoch // num_batches_to_accumulate, 1)
    scheduler_warmup_accum_steps = max(50, int(accum_steps_in_epoch * 0.5))  # Half epoch warmup
    scheduler_accum_steps_in_first_cycle = scheduler_warmup_accum_steps * 10  # Shorter cycles
    
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

def log_training_info(rank, model, trnds, valdl, trndl, params, batch_size, world_size):
    """Log training information."""
    if rank != 0:
        return
    
    # Count only trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    logging.info(f"=== Adapter Training Configuration ===")
    logging.info(f"Total model parameters: {total_params/1e6:.2f} million")
    logging.info(f"Trainable adapter parameters: {trainable_params/1e6:.2f} million")
    logging.info(f"Training samples: {len(trnds)}")
    logging.info(f"Batch size: {batch_size}, World size: {world_size}")
    logging.info(f"Effective accumulation batch size: {params['effective_accum_batch_size']}")
    logging.info(f"Training epochs: {params['training_epochs']}")
    logging.info(f"Training max steps: {params['training_max_steps']}")
    logging.info(f"Batches in epoch: {params['batches_in_epoch']}")
    logging.info(f"Scheduler warmup steps: {params['scheduler_warmup_accum_steps']}")
    logging.info(f"Train batches: {len(trndl)}")
    logging.info(f"Validation batches: {len(valdl)}")
    logging.info(f"Number of benchmark properties: {model.num_tasks}")

def main(rank, world_size):
    """Main training function."""
    init_logging(rank)
    
    # Setup directories
    outdir = pathlib.Path("cache/train_adapters_all_props")
    cachedir = outdir / "cache"
    cachedir.mkdir(parents=True, exist_ok=True)
    modeldir = outdir / "models"
    metricsdir = outdir / "metrics"
    
    logging.info(f"Rank {rank} starting setup.")
    
    # Load tokenizer
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
    
    # Load benchmark properties
    bp = pd.read_parquet("cache/get_benchmark_properties/benchmark_properties.parquet")
    benchmark_properties = list(bp['property_token'].unique())
    
    logging.info(f"Loaded {len(benchmark_properties)} benchmark properties")
    
    # Create model and optimizer
    min_lr = 1e-4
    max_lr = 1e-3  # Lower learning rate for adapters
    # Note: 3_1_1 saves MultitaskTransformer under MoE path due to incorrect save path
    shared_base_path = "cache/train_multitask_transformer_parallel/models/MoE/best_loss"
    
    model, optimizer = create_model_and_optimizer(
        tokenizer, benchmark_properties, shared_base_path, base_lr=min_lr
    )

    # Load datasets - use same structure as 3_1_1 but don't filter properties
    logging.info("Building training/validation datasets")
    allpaths = list(pathlib.Path("cache/build_tensordataset/multitask_tensors/trn").glob("*.pt"))
    tstpaths = list(pathlib.Path("cache/build_tensordataset/multitask_tensors/hld").glob("*.pt"))
    trnpaths = [path for path in allpaths if path not in tstpaths]
    
    # Use all properties (no filtering) - let find_unused_parameters handle unused adapters
    batch_size = 1000
    # trnds, valds = create_datasets(tokenizer, rank)    
    trnds, valds = create_testing_datasets(tokenizer)  # For quick testing
    trndl, valdl = create_dataloaders(tokenizer, trnds, valds, batch_size, world_size, rank)

    # Calculate training parameters
    logging.info("Calculating training params")
    params = calculate_training_params(trnds, valds, batch_size, world_size)
    
    scheduler = WarmupCosineThenPlateau(
        optimizer,
        warmup_steps=params["scheduler_warmup_accum_steps"],
        cosine_cycle_length=params["scheduler_accum_steps_in_first_cycle"],
        min_lr=min_lr,
        max_lr=max_lr,
        plateau_mode="min",
        plateau_factor=0.7,
        plateau_patience=3,
        plateau_min_lr=1e-6,
        plateau_verbose=True,
    )

    # Log training info
    log_training_info(rank, model, trnds, valdl, trndl, params, batch_size, world_size)

    stratified_eval = StratifiedGroupEvaluator(tokenizer=tokenizer, rank=rank)
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
        ema_alpha=0.1,
        
        # Adjust for large batch size
        skip_initial_batches=10,
        sampling_warmup=20,
        
        label_smoothing=0.15,
        unsampled_property_weight=0.0
    )

    # Setup trainer
    trainer.set_validation_dataloader(valdl)
    trainer.set_model_savepath(modeldir / "adapter_model")
    trainer.set_metrics_file(metricsdir / "adapter_loss.tsv", overwrite=True)
    
    logging.info(f"Rank {rank} starting adapter training.")
    trainer.start()

if __name__ == "__main__":
    # Get distributed training parameters
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # Setup distributed training
    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="gloo",
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
