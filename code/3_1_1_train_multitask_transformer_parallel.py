#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHONPATH=./ CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nproc-per-node=8 --master-port=29500 code/3_1_1_train_multitask_transformer_parallel.py 2> cache/train_multitask_transformer_parallel/logs/err.log

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
from cvae.models.datasets import InMemorySelfiesPropertiesValuesDataset, BalancedSelfiesPropertiesValuesDataset, SimplePropertyMappedDataset, DynamicBalancedSelfiesDataset
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
    # torch.set_float32_matmul_precision('high')

<<<<<<< HEAD
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
    # mt.ToxTransformer(tokenizer)
    
<<<<<<< Updated upstream
    # model = torch.compile(model)
    # model = mt.ToxTransformer(tokenizer)
=======
    # model = torch.compile(model, mode='max-autotune')
>>>>>>> Stashed changes
=======
    # new mte
    # 2025-09-17 07:47:38,205 - INFO - root - 📊 STRATIFIED EVAL - Loss: 0.4812, AUC: 0.8715, BAC: 0.8098
    # 2025-09-17 07:47:38,208 - INFO - root - ✅ EVAL COMPLETE [Step 73835] - Loss: 0.4812 (best: 0.4692), AUC: 0.8715 (best: 0.8738), BAC: 0.8098, LR: 1.00e-05
    
    # V0.99 AUC .8477 Loss .4365
    # config = mte.MultitaskEncoderConfig(
    #     hdim=512,
    #     nhead=16,
    #     ff_mult=3,
    #     num_layers=8,
    #     # activation='gelu',
    #     dropout_rate=0.1, 
    #     attention_dropout=0.1,
    #     layer_dropout=0.1,
    #     output_size=2,
    # )

    config = mte.MultitaskEncoderConfig(
        hdim=768,
        nhead=24,
        ff_mult=3,
        num_layers=12,
        # activation='gelu',
        dropout_rate=0.1, 
        attention_dropout=0.1,
        layer_dropout=0.1,
        output_size=2,
    )

    model = mte.MultitaskEncoder(tokenizer=tokenizer, config=config)
    # model = mte.MultitaskEncoder.load('cache/train_multitask_transformer_parallel/models/me_roundrobin/best_loss')
    
    # model = torch.compile(model, mode='reduce-overhead', fullgraph=False, dynamic=True)
    model = torch.compile(model, mode='default', fullgraph=True, dynamic=False)
>>>>>>> 887eb71 (major updates and catch up)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=base_lr, betas=(0.9, 0.98), weight_decay=1e-3, eps=1e-8, fused=False  
    )
    
    return model, optimizer

def create_dataloaders(tokenizer, trnds, valds, batch_size, world_size, rank):
    """Create training and validation dataloaders."""
    # Reduced settings to prevent memory leaks
    cpus_per_rank = max(2, os.cpu_count() // world_size)
    train_workers = max(2, min(6, cpus_per_rank))  # Reduced
    val_workers = max(1, min(4, cpus_per_rank))    # Reduced
    prefetch_factor = 10  # Significantly reduced from 40

    # Create dataloaders
    trndl = torch.utils.data.DataLoader(
        trnds, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=train_workers,
        pin_memory=True,          # Keep this for speed
        persistent_workers=True,  # Keep this for speed  
        prefetch_factor=prefetch_factor,  # Much lower
        # sampler=custom_sampler.FastDistributedSampler(
        #     trnds, num_replicas=world_size, rank=rank, shuffle=True, seed=42
        # )
        # sampler=torch.utils.data.distributed.DistributedSampler(
        #     trnds, num_replicas=world_size, rank=rank, drop_last=False, shuffle=True
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
        # sampler=custom_sampler.FastDistributedSampler(
        #     valds, num_replicas=world_size, rank=rank, shuffle=True, seed=42
        # )
        # sampler=torch.utils.data.distributed.DistributedSampler(
        #     valds, num_replicas=world_size, rank=rank, drop_last=False, shuffle=True
        # )
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
    training_epochs = 1000
    target_effective_batch_size = batch_size * world_size * 2

    # Calculate steps and accumulation
    batches_in_epoch = max(len(trnds) // (batch_size * world_size), 1)
    training_max_steps = training_epochs * batches_in_epoch
    num_batches_to_accumulate = max(target_effective_batch_size // (batch_size * world_size), 1)
    effective_accum_batch_size = batch_size * world_size * num_batches_to_accumulate
    
    # validation params
    desired_validation_samples = len(all_properties) * 2 * 100 # 100 samples per property, positive and negative
    validation_batches = max(desired_validation_samples // (batch_size * world_size), 1)
    # validation_batches = len(valds)
    

    # Scheduler parameters
    accum_steps_in_epoch = max(batches_in_epoch // num_batches_to_accumulate, 1)
    scheduler_warmup_accum_steps = 4000
    scheduler_accum_steps_in_first_cycle = 20_000
    
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
    outdir = pathlib.Path("cache/train_multitask_transformer_parallel")
    cachedir = outdir / "cache"
    cachedir.mkdir(parents=True, exist_ok=True)
    modeldir = outdir / "models"
    metricsdir = outdir / "metrics"

    logging.info(f"Rank {rank} starting setup.")

    # Synchronize all ranks before proceeding
    dist.barrier()

    # Load tokenizer
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

    dist.barrier()
    
    # Dataset configuration
    all_properties = list(range(tokenizer.num_assays))
    trnpaths = list(pathlib.Path("cache/build_tensordataset/multitask_tensors/trn").glob("*.pt"))
    tstpaths = list(pathlib.Path("cache/build_tensordataset/multitask_tensors/hld").glob("*.pt"))
    
    # took 1528 steps to get to 92% AUC
    # tmppaths = trnpaths
    # trnds = InMemorySelfiesPropertiesValuesDataset(tmppaths[:30], tokenizer, nprops=5)
    # valds = InMemorySelfiesPropertiesValuesDataset(tmppaths[:10], tokenizer, nprops=1)
    
    # trnds = InMemorySelfiesPropertiesValuesDataset(paths=trnpaths, tokenizer=tokenizer, nprops=5)
    trnds = SimplePropertyMappedDataset(
        paths=trnpaths,
        tokenizer=tokenizer,
        target_properties=all_properties,
        nprops=5,
        seed=rank
    )
    # # logging.info(f"Training dataset loaded: {len(trnds)} samples")

    # # logging.info("Loading validation dataset")
    valds = InMemorySelfiesPropertiesValuesDataset(tokenizer=tokenizer, paths=tstpaths, nprops=1)
    # valds = SimplePropertyMappedDataset(
    #     paths=tstpaths,
    #     tokenizer=tokenizer,
    #     target_properties=all_properties,
    #     nprops=1,
    #     seed=rank
    # )
    logging.info(f"Validation dataset loaded: {len(valds)} samples")

    logging.info("Getting optimal batch size...")
    
    # Synchronize before creating model
    dist.barrier()

    # Create model and optimizer
<<<<<<< HEAD
    model, optimizer = create_model_and_optimizer(tokenizer)
    
    # Training configuration
<<<<<<< Updated upstream
    batch_size = 32 * 13
=======
    batch_size = 32 * 5
>>>>>>> Stashed changes
=======
    min_lr = 1e-4
    max_lr = 5e-4
    model, optimizer = create_model_and_optimizer(tokenizer, base_lr=min_lr)

    dist.barrier()

    batch_size = 1000 # lastrun
    # batch_size = 1800 # gelurun
    # batch_size = find_optimal_batch_size(
    #     model=model, 
    #     dataset=trnds, 
    #     min_batch_size=32, 
    #     max_batch_size=4096, 
    #     target_memory_percent=80.0
    # )[0]
>>>>>>> 887eb71 (major updates and catch up)
    
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
        plateau_patience=5000,
        plateau_min_lr=1e-4,
        plateau_verbose=True,
    )
    
    # Log training info
    log_training_info(rank, model, trnds, valdl, trndl, params, batch_size, world_size)

    # Setup evaluator and trainer
    # stratified_eval = StratifiedEvaluator(num_tasks=len(tokenizer.assay_indexes()), rank=rank)
    stratified_eval = StratifiedGroupEvaluator(tokenizer=tokenizer, rank=rank)
    
    desired_training_samples_before_eval = 10_000_000
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
        max_score=1.0, # was 10
        min_score=1.0, # was .1
        skip_initial_batches=5,
        property_sample_rate=.66,
        sample_bias_strength=2.0,
        ema_alpha=.1,
        label_smoothing=.15,
    )

    # Setup trainer
    trainer.set_validation_dataloader(valdl)
    trainer.set_model_savepath(modeldir / "me_roundrobin")
    trainer.set_metrics_file(metricsdir / "multitask_loss.tsv", overwrite=True)
    
    logging.info(f"Rank {rank} starting training.")
    trainer.start()

if __name__ == "__main__":
    # Read ranks from env set by torchrun
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", "0"))
    RANK = int(os.environ.get("RANK", "0"))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))

<<<<<<< HEAD
<<<<<<< Updated upstream
    # enable_profiling = rank == 0  # Set to True to enable
=======
    # enable_profiling = rank == 0 and False  # Set to True to enable
>>>>>>> Stashed changes
    
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
=======
    # Sanity: make sure CUDA is actually available and we have enough devices
    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. Check that you're on a GPU machine, the driver is loaded, "
            "and CUDA-visible inside the process/container (nvidia-smi, NVIDIA_VISIBLE_DEVICES)."
        )
>>>>>>> 887eb71 (major updates and catch up)

    num_gpus = torch.cuda.device_count()
    if num_gpus == 0:
        raise RuntimeError("No CUDA devices visible (torch.cuda.device_count()==0).")
    if LOCAL_RANK >= num_gpus:
        raise RuntimeError(
            f"LOCAL_RANK {LOCAL_RANK} >= visible GPUs {num_gpus}. "
            "Fix --nproc-per-node or CUDA_VISIBLE_DEVICES."
        )

    # Bind this process to its local GPU
    torch.cuda.set_device(LOCAL_RANK)

    # Initialize the process group (using Gloo to avoid NCCL segfaults)
    dist.init_process_group(
        backend="gloo",  # Changed from nccl due to segfaults
        init_method="env://",
        rank=RANK,
        world_size=WORLD_SIZE,
        timeout=datetime.timedelta(minutes=30),
    )

    print(f"[Init] RANK={RANK} LOCAL_RANK={LOCAL_RANK} WORLD_SIZE={WORLD_SIZE} GPU={LOCAL_RANK}")

    try:
        main(rank=RANK, world_size=WORLD_SIZE)
    finally:
        dist.barrier()
        dist.destroy_process_group()