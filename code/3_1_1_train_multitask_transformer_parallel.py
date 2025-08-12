#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHONPATH=./ CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nproc-per-node=8 --master-port=29500 code/3_1_1_train_multitask_transformer_parallel.py 2> cache/train_multitask_transformer_parallel/logs/err.log

import os
import pathlib
import logging
import datetime

import torch
import torch.utils.data
import torch.distributed as dist
import pandas as pd

import cvae.tokenizer
import cvae.models.multitask_transformer as mt
from helper.trainer.selfies_properties_values_trainer import SelfiesPropertiesValuesTrainer
from helper.scheduler.warmup_cosine_trainer import WarmupCosineScheduler
from helper.trainer.gradnorm_trainer import GradNormTrainer
from helper.trainer.invfreq_trainer import InverseFrequencyWeightedTrainer
from cvae.models.datasets import InMemorySelfiesPropertiesValuesDataset

# Environment variables setup
os.environ.update({
    'MKL_THREADING_LAYER': 'GNU',
    'MKL_SERVICE_FORCE_INTEL': '0',
    'OMP_NUM_THREADS': '1',
    'MKL_NUM_THREADS': '1',
    'PYTHONWARNINGS': 'ignore::FutureWarning',
    'TORCH_NCCL_ASYNC_ERROR_HANDLING': '1',
    'NCCL_IB_DISABLE': '0',
    'NCCL_P2P_DISABLE': '0'
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

def create_model_and_optimizer(tokenizer):
    """Create model and optimizer."""
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # Create model
    model = mt.MultitaskTransformer(
        tokenizer=tokenizer, 
        hdim=256, 
        nhead=8, 
        num_layers=48, 
        ff_mult=4,
        dropout_rate=0.1, 
        output_size=2
    )
    
    # model = torch.compile(model)
    # model = mt.ToxTransformer(tokenizer)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=1e-3, 
        betas=(0.9, 0.99), 
        weight_decay=1e-2
    )
    
    return model, optimizer

def create_dataloaders(tokenizer, batch_size, world_size, rank):
    """Create training and validation dataloaders."""
    # Load datasets
    # dataset_path = "cache/build_tensordataset/multitask_tensors/tmp"
    # trnds = InMemorySelfiesPropertiesValuesDataset(dataset_path, tokenizer, nprops=5)
    # valds = trnds  # Using same dataset for now
    trnds = InMemorySelfiesPropertiesValuesDataset("cache/build_tensordataset/multitask_tensors/trn", tokenizer, nprops=5)
    valds = InMemorySelfiesPropertiesValuesDataset("cache/build_tensordataset/multitask_tensors/tst", tokenizer, nprops=5)
    
    # Calculate workers
    cpus_per_rank = max(2, os.cpu_count() // world_size)
    train_workers = max(2, min(32, cpus_per_rank))
    val_workers = max(1, min(20, cpus_per_rank))
    prefetch_factor = 2

    # Create dataloaders
    trndl = torch.utils.data.DataLoader(
        trnds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=train_workers,
        pin_memory=True, 
        persistent_workers=True, 
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
        pin_memory=True, 
        persistent_workers=True, 
        prefetch_factor=prefetch_factor,
        sampler=torch.utils.data.distributed.DistributedSampler(
            valds, num_replicas=world_size, rank=rank, drop_last=True, shuffle=True
        )
    )
    
    return trndl, valdl, trnds

def calculate_training_params(trnds, batch_size, world_size):
    """Calculate training parameters."""
    # Training configuration
    training_epochs = 10
    target_effective_batch_size = 16000 * 2 # about 100 updates per epoch

    # Calculate steps and accumulation
    batches_in_epoch = max(len(trnds) // (batch_size * world_size), 1)
    training_max_steps = training_epochs * batches_in_epoch
    num_batches_to_accumulate = max(target_effective_batch_size // (batch_size * world_size), 1)
    effective_accum_batch_size = batch_size * world_size * num_batches_to_accumulate
    
    # Scheduler parameters
    accum_steps_in_epoch = max(batches_in_epoch // num_batches_to_accumulate, 1)
    scheduler_warmup_accum_steps = max(100, int(accum_steps_in_epoch * training_epochs * 0.05))
    scheduler_accum_steps_in_first_cycle = accum_steps_in_epoch * 3
    
    return {
        'training_max_steps': training_max_steps,
        'effective_accum_batch_size': effective_accum_batch_size,
        'scheduler_warmup_accum_steps': scheduler_warmup_accum_steps,
        'scheduler_accum_steps_in_first_cycle': scheduler_accum_steps_in_first_cycle,
        'batches_in_epoch': batches_in_epoch,
        'accum_steps_in_epoch': accum_steps_in_epoch,
        'training_epochs': training_epochs
    }

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

def main(rank, world_size):
    """Main training function."""
    init_logging(rank)
    
    # Setup directories
    outdir = pathlib.Path("cache/train_multitask_transformer_parallel")
    cachedir = outdir / "cache"
    cachedir.mkdir(parents=True, exist_ok=True)
    modeldir = outdir / "models"
    metricsdir = outdir / "metrics"
    
    logging.info(f"Rank {rank} starting setup.")
    
    # Load tokenizer
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
    
    # Create model and optimizer
    model, optimizer = create_model_and_optimizer(tokenizer)
    
    # Training configuration
    batch_size = 32 * 13
    
    # Create dataloaders
    trndl, valdl, trnds = create_dataloaders(tokenizer, batch_size, world_size, rank)
    
    # Calculate training parameters
    params = calculate_training_params(trnds, batch_size, world_size)
    
    # Log training info
    log_training_info(rank, model, trnds, valdl, trndl, params, batch_size, world_size)
    
    # Create scheduler
    base_lr = 1e-3
    base_batch_size = 8000
    lr_scaling_factor = params['effective_accum_batch_size'] / base_batch_size
    scaled_max_lr = base_lr * lr_scaling_factor
    scheduler = WarmupCosineScheduler(
        optimizer=optimizer,
        warmup_steps=params['scheduler_warmup_accum_steps'],
        cosine_cycle_length=params['scheduler_accum_steps_in_first_cycle'],
        min_lr=1e-6,
        max_lr=scaled_max_lr,
        cosine_t_mult=2
    )
    
    # Create trainer
    # trainer = SelfiesPropertiesValuesTrainer(
    #     model=model,
    #     rank=rank,
    #     tokenizer=tokenizer,
    #     trn_iterator=trndl,
    #     batch_size=batch_size,
    #     scheduler=scheduler,
    #     max_steps=params['training_max_steps'],
    #     effective_accum_batch_size=params['effective_accum_batch_size'],
    #     eval_samples=2000,
    #     first_eval=5,
    #     eval_every=1000,
    #     find_unused_parameters=True
    # )


    trainer = InverseFrequencyWeightedTrainer(
        model=model,
        rank=rank,
        tokenizer=tokenizer,
        trn_iterator=trndl,
        batch_size=batch_size,
        scheduler=scheduler,
        max_steps=params['training_max_steps'],
        effective_accum_batch_size=params['effective_accum_batch_size'],
        eval_samples=2000,
        first_eval=5,
        eval_every=1000,
        find_unused_parameters=True,
        ema_decay=0.99,     # optional
        ema_eps=1e-6,       # optional
        max_weight=1000.0,
        batch_level_weighting=True
    )

    # trainer = GradNormTrainer(
    #     model=model,
    #     rank=rank,
    #     tokenizer=tokenizer,
    #     trn_iterator=trndl,
    #     batch_size=batch_size,
    #     scheduler=scheduler,
    #     max_steps=params['training_max_steps'],
    #     effective_accum_batch_size=params['effective_accum_batch_size'],
    #     eval_samples=2000,
    #     first_eval=5,
    #     eval_every=1000,
    #     find_unused_parameters=True,
    #     gradnorm_cfg=dict(
    #         learning_rate=1e-4,           # weight update LR for GradNorm
    #         restoring_force_alpha=0.0     # good default for sparse ~4k tasks
    #     )
    # )
    
    # Setup trainer
    trainer.set_validation_dataloader(valdl)
    trainer.set_model_savepath(modeldir / "multitask_transformer")
    trainer.set_metrics_file(metricsdir / "multitask_loss.tsv", overwrite=True)
    
    # Start training
    logging.info(f"warming up trainer")
    warmup_loader = torch.utils.data.DataLoader(
        trnds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=2,
        pin_memory=True, 
        persistent_workers=False
    )
    trainer.warmup_frequencies(warmup_steps=len(warmup_loader), dataloader=warmup_loader, cache_path=str(cachedir / "inv_freq_cache"))
    logging.info(f"Rank {rank} starting training.")
    trainer.start()

if __name__ == "__main__":
    # Get distributed training parameters
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])

    # enable_profiling = rank == 0  # Set to True to enable
    
    # if enable_profiling:
    #     from torch.profiler import profile, record_function, ProfilerActivity
    #     prof = profile(
    #         activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
    #         schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=1),
    #         on_trace_ready=torch.profiler.tensorboard_trace_handler('./profiler_logs'),
    #         record_shapes=True,
    #         with_stack=True
    #     )
    #     prof.start()

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