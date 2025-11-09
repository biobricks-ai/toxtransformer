#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
rm -rf cache/train_adapters_all_props/
for SPLIT in {0..3}; do
  LOGDIR="cache/train_adapters_all_props/logs/split_${SPLIT}"
  mkdir -p "$LOGDIR"
  
  SPLIT=$SPLIT LOGDIR="$LOGDIR" PYTHONPATH=./ CUDA_LAUNCH_BLOCKING=1 \
  torchrun --standalone --nproc-per-node=8 --master-port=29500 \
    code/3_1_2_train_adapters_all_props.py \
    1> "${LOGDIR}/out.log" \
    2> "${LOGDIR}/err.log"
done
./slackmsg "Finished training adapters on all benchmark properties for all splits."
"""

# Increase file descriptor limit
import resource
resource.setrlimit(resource.RLIMIT_NOFILE, (65536, 65536))

import sys
import os
import pathlib
import logging
import datetime

import torch
import torch.utils.data
import torch.distributed as dist
import pandas as pd

import cvae.tokenizer
from cvae.models.adapter_model import SimpleAdapterModel
from cvae.models.multitask_encoder import MultitaskEncoder
from helper.trainer.selfies_properties_values_trainer import SelfiesPropertiesValuesTrainer
from helper.scheduler.warmup_cosine_trainer import WarmupCosineScheduler, WarmupCosineThenPlateau
from helper.trainer.invfreq_trainer import InverseFrequencyWeightedTrainer
from helper.trainer import AccumulatedStratifiedPropertyWeightedTrainer
from helper.trainer.evaluator import StratifiedGroupEvaluator

from cvae.models.datasets import InMemorySelfiesPropertiesValuesDataset, SimplePropertyMappedDataset
from code.helper.utils import find_optimal_batch_size

import math
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

def _resolve_run_dirs():
    """Resolve per-run directories from LOGDIR/SPLIT and ensure they exist."""
    base = pathlib.Path(
        os.getenv("LOGDIR",
                  f"cache/train_adapters_all_props/logs/split_{os.getenv('SPLIT','0')}")
    )
    models = base / "models"
    metrics = base / "metrics"
    for d in (base, models, metrics):
        d.mkdir(parents=True, exist_ok=True)
    return base, models, metrics

def init_logging(rank):
    """Initialize logging for rank 0 only into LOGDIR."""
    if rank != 0:
        return
    logdir, _, _ = _resolve_run_dirs()
    logfile = logdir / "train_adapters_all_props.log"
    logdir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        filename=logfile,
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        filemode="w",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logging.getLogger().addHandler(console)
    logging.info("Logging initialized in %s", str(logfile))

def create_model_and_optimizer(tokenizer, num_benchmark_properties, shared_base_path, base_lr):
    """Create adapter model and optimizer."""
    # Enable optimizations
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision('high')

    # Load the pre-trained encoder
    encoder = MultitaskEncoder.load(shared_base_path)
    
    # Create adapter model with all benchmark properties
    model = SimpleAdapterModel(
        multitask_encoder=encoder,
        num_properties=num_benchmark_properties,
    )

    model = torch.compile(model, mode='default', fullgraph=True, dynamic=False)

    # Only optimize the adapter parameters (encoder is frozen)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], 
        lr=base_lr, 
        betas=(0.9, 0.99),
    )
    
    return model, optimizer

def create_datasets(tokenizer, rank, split, world_size, nprops=20, min_samples=8):
    """Create datasets for a specific split.
    - Each training file is assigned to exactly min_samples different ranks (no duplicates per rank)
    - Test files are divided across ranks WITHOUT overlap so each rank gets a unique subset.
    - Each rank receives a disjoint subset of target_properties (round-robin by property index).
    """
    assert min_samples <= world_size, f"min_samples ({min_samples}) must be <= world_size ({world_size})"
    
    # Use bootstrap split data (matching base model training)
    base_path = pathlib.Path(f"cache/build_tensordataset/bootstrap/split_{split}")
    trnpaths_all = sorted((base_path / "train").glob("*.pt"))
    tstpaths_all = sorted((base_path / "test").glob("*.pt"))

    # Deterministic shuffle for train for each split
    rng = random.Random(split)
    rng.shuffle(trnpaths_all)

    # Assign each file to exactly min_samples different ranks
    trnpaths = []
    for file_idx, filepath in enumerate(trnpaths_all):
        # Round-robin assignment: file i goes to ranks [i*min_samples, i*min_samples+1, ..., i*min_samples+min_samples-1] mod world_size
        start_rank = (file_idx * min_samples) % world_size
        assigned_ranks = [(start_rank + j) % world_size for j in range(min_samples)]
        
        if rank in assigned_ranks:
            trnpaths.append(filepath)
    
    # Shuffle paths for this rank
    rng.shuffle(trnpaths)

    # Split test dataset into non-overlapping chunks
    n_tst = len(tstpaths_all)
    chunk_tst = math.ceil(n_tst / world_size)
    start_tst = rank * chunk_tst
    end_tst = min(n_tst, start_tst + chunk_tst)
    tstpaths = tstpaths_all[start_tst:end_tst]

    # Partition properties across ranks (round-robin by property index) so each rank gets a different subset
    num_props = tokenizer.num_assays
    all_props = list(range(num_props))
    assigned_properties = [p for p in all_props if (p % world_size) == rank]

    # Ensure sensible nprops for datasets (can't request more properties than assigned)
    train_nprops = min(nprops, max(1, len(assigned_properties)))
    val_nprops = min(1, max(1, len(assigned_properties)))

    logging.info(
        f"Split {split} | Rank {rank}/{world_size} | "
        f"{len(trnpaths)} training files (from {len(trnpaths_all)}) | "
        f"{len(tstpaths)} test files (of {n_tst}) | "
        f"{len(assigned_properties)} assigned properties: {assigned_properties}"
    )

    trnds = SimplePropertyMappedDataset(
        paths=trnpaths,
        tokenizer=tokenizer,
        target_properties=assigned_properties,
        nprops=train_nprops,
        seed=rank + split * 1000,
    )

    valds = SimplePropertyMappedDataset(
        paths=tstpaths,
        tokenizer=tokenizer,
        target_properties=assigned_properties,
        nprops=val_nprops,
    )

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

def calculate_training_params(trnds, valds, batch_size, world_size, all_properties):
    """Calculate training parameters."""

    # Calculate steps and accumulation
    target_effective_batch_size = batch_size * world_size
    num_batches_to_accumulate = max(target_effective_batch_size // (batch_size * world_size), 1)
    effective_accum_batch_size = batch_size * world_size * num_batches_to_accumulate
    
    # validation params
    desired_validation_samples = len(all_properties) * 2 * 100 # 100 samples per property, positive and negative
    validation_batches = max(desired_validation_samples // (batch_size * world_size), 1)

    # Scheduler parameters
    scheduler_warmup_accum_steps = 1000
    scheduler_accum_steps_in_first_cycle = 30_000
    
    return {
        'training_max_steps': 20_000,
        'effective_accum_batch_size': effective_accum_batch_size,
        'scheduler_warmup_accum_steps': scheduler_warmup_accum_steps,
        'scheduler_accum_steps_in_first_cycle': scheduler_accum_steps_in_first_cycle,
        'validation_batches': validation_batches,
    }

def log_training_info(rank, model, trnds, valdl, trndl, params, batch_size, world_size, split):
    """Log training information."""
    if rank != 0:
        return
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logging.info(f"=== Training Configuration for Split {split} ===")
    logging.info(f"Model parameters: {num_params/1e6:.2f} million")
    logging.info(f"Training samples: {len(trnds)}")
    logging.info(f"Batch size: {batch_size}, World size: {world_size}")
    logging.info(f"Effective accumulation batch size: {params['effective_accum_batch_size']}")
    logging.info(f"Training max steps: {params['training_max_steps']}")
    logging.info(f"Scheduler warmup steps: {params['scheduler_warmup_accum_steps']}")
    logging.info(f"Train batches: {len(trndl)}")
    logging.info(f"Validation batches: {len(valdl)}")
    logging.info(f"Warmup as % of total: {(params['scheduler_warmup_accum_steps'] / params['training_max_steps']) * 100:.2f}%")

def find_best_checkpoint(base_model_dir, split):
    """Find the best checkpoint from base model training.
    
    Looks for:
    1. best_loss checkpoint (preferred)
    2. Latest checkpoint_step_* if best_loss doesn't exist
    """
    # Expected location based on base model training script
    base_path = pathlib.Path(f"cache/train_multitask_transformer_parallel/logs/split_{split}/models/me_roundrobin_property_dropout_V3")
    
    # First try best_loss
    best_loss_path = base_path / "best_loss"
    if best_loss_path.exists():
        logging.info(f"Found best_loss checkpoint at {best_loss_path}")
        return str(best_loss_path)
    
    # Otherwise find latest checkpoint
    checkpoints = list(base_path.glob("checkpoint_step_*"))
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints found in {base_path}")
    
    # Sort by step number
    checkpoints.sort(key=lambda p: int(p.name.split("_")[-1]))
    latest = checkpoints[-1]
    logging.info(f"Using latest checkpoint: {latest}")
    return str(latest)

def main(rank, world_size, split):
    """Main training function."""
    init_logging(rank)
    logging.info(f"Rank {rank} starting setup for split {split}.")

    # Use LOGDIR for outputs
    logdir, modeldir, metricsdir = _resolve_run_dirs()
    logging.info("Run directories: logdir=%s models=%s metrics=%s", logdir, modeldir, metricsdir)

    # Load tokenizer
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
    
    # Create datasets
    trnds, valds = create_datasets(tokenizer, rank, split, world_size)

    # Create dataloaders
    logging.info("Creating dataloaders...")
    batch_size = 1000
    trndl, valdl = create_dataloaders(tokenizer, trnds, valds, batch_size, world_size, rank)

    # Calculate training parameters
    logging.info("Calculating training params...")
    params = calculate_training_params(trnds, valds, batch_size, world_size, all_properties=range(tokenizer.num_assays))

    # Find best checkpoint from base model training
    shared_base_path = find_best_checkpoint(
        base_model_dir=f"cache/train_multitask_transformer_parallel/logs/split_{split}/models",
        split=split
    )
    logging.info(f"Loading base model from: {shared_base_path}")
    
    # Create model, optimizer and scheduler
    min_lr = 1e-4
    max_lr = 1e-4
    
    model, optimizer = create_model_and_optimizer(
        tokenizer, 
        tokenizer.num_assays, 
        shared_base_path, 
        base_lr=min_lr
    )
    
    scheduler = WarmupCosineThenPlateau(
        optimizer,
        warmup_steps=params["scheduler_warmup_accum_steps"],
        cosine_cycle_length=params["scheduler_accum_steps_in_first_cycle"],
        warmup_start_lr=min_lr,
        warmup_end_lr=max_lr,
        cosine_start_lr=max_lr,
        cosine_end_lr=min_lr,
        plateau_start_lr=min_lr,
        plateau_end_lr=min_lr / 10,
        plateau_mode="min",
        plateau_factor=0.8,
        plateau_patience=200,
        plateau_verbose=True,
    )
    dist.barrier()

    # Log training info
    log_training_info(rank, model, trnds, valdl, trndl, params, batch_size, world_size, split)

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
        eval_every=500,
        find_unused_parameters=False,
        evaluator=stratified_eval,
        
        # Minimal accumulation for adapters
        acc_coalesce_every=1,
        
        # Moderate weighting
        max_score=1.0,
        min_score=1.0,
        
        # Balanced sampling
        property_sample_rate=1.0,
        sample_bias_strength=0.0,
        
        # Faster EMA for adapters
        ema_alpha=0.9,
        
        # Adjust for adapter training
        skip_initial_batches=0,
        sampling_warmup=20,

        label_smoothing=label_smoothing,
        unsampled_property_weight=0.0
    )

    # Setup trainer
    trainer.set_validation_dataloader(valdl)
    trainer.set_model_savepath(modeldir / "adapter_model")
    trainer.set_metrics_file(metricsdir / "adapter_loss.tsv", overwrite=True)
    
    logging.info(f"Rank {rank} starting adapter training for split {split}.")
    trainer.start()

if __name__ == "__main__":
    RANK = int(os.environ.get("RANK", "0"))
    WORLD_SIZE = int(os.environ.get("WORLD_SIZE", "1"))
    SPLIT = int(os.environ.get("SPLIT", "0"))
    LOCAL_RANK = int(os.environ.get("LOCAL_RANK", str(RANK)))

    # Bind this process to its local GPU
    torch.cuda.set_device(LOCAL_RANK)

    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=RANK,
        world_size=WORLD_SIZE,
        timeout=datetime.timedelta(minutes=30),
    )

    print(f"[Init] RANK={RANK} LOCAL_RANK={LOCAL_RANK} WORLD_SIZE={WORLD_SIZE} GPU={LOCAL_RANK}")
    main(rank=RANK, world_size=WORLD_SIZE, split=SPLIT)