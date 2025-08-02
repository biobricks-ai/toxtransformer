#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHONPATH=./ CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nproc-per-node=8 --master-port=29500 code/3_1_2_train_adapters.py 2> cache/train_adapters/logs/err.log
# PYTHONPATH=./ CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nproc-per-node=1 --master-port=29500 code/3_1_2_train_adapters.py 2> cache/train_adapters/logs/err.log

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
        logdir = pathlib.Path("cache/train_adapters/logs")
        logdir.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(filename=logdir / "train_adapters.log", level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
        logging.info("Logging initialized.")

def main(rank, world_size):
     
    # torch.backends.cuda.matmul.allow_tf32 = True
    # torch.backends.cudnn.allow_tf32 = True
    # torch.backends.cudnn.benchmark = True
    init_logging(rank)

    outdir = pathlib.Path("cache/train_adapters")
    modeldir = outdir / "models"
    modeldir.mkdir(exist_ok=True, parents=True)

    metricsdir = outdir / "metrics"
    metricsdir.mkdir(exist_ok=True, parents=True)
    
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
    
    cpus_per_rank = max(2, os.cpu_count() // world_size)
    train_workers = max(2, min(32, cpus_per_rank))
    val_workers = max(1, min(20, cpus_per_rank))

    bp = pd.read_parquet("cache/get_benchmark_properties/benchmark_properties.parquet")
    # bp = bp[~bp['source'].isin(['pubchem','bindingdb'])]
    
    benchmark_properties = list(bp['property_token'].unique())
    tox21_props = list(bp[bp['source'].isin(['Tox21'])]['property_token'].unique().tolist())

    model : me.MoE = me.MoE.load("cache/train_multitask_transformer_parallel/models/step_16000")
    model.save(modeldir / "moe")
    model.freeze_shared_embedding()
    model.freeze_main_experts()

    # adapter_props = bp[~bp['source'].isin(['pubchem','bindingdb','ctdbase','reach','chembl','toxcast','ice','toxvaldb'])]['property_token'].unique().tolist()
    adapter_props = tox21_props
    for prop in adapter_props:
        _ = model.add_boosting_expert(prop, expert_layers=2, expert_nhead=2, expert_dim_feedforward=256, expert_dropout_rate=0.3)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    batch_size = 32

    # tmp = InMemorySequenceShiftDataset("cache/build_tensordataset/multitask_tensors/tmp", tokenizer, nprops=10)
    # trnds = tmp
    # valds = tmp

    # rotate read order of training tensors
    trnpaths = sorted(pathlib.Path("cache/merge_tensordataset/multitask_tensors/trn").glob("*.pt"))
    trnpaths = trnpaths[rank:] + trnpaths[:rank]
    trnds = InMemorySequenceShiftDataset.from_cache_or_create(
        trnpaths, tokenizer, nprops=10, assay_filter=tox21_props,
        cache_path= outdir / "inmemory_dataset_cache" / "trn_dataset.pt")

    trndl = torch.utils.data.DataLoader(
        trnds, batch_size=batch_size, shuffle=False, num_workers=train_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=100,
        sampler=torch.utils.data.distributed.DistributedSampler(trnds, num_replicas=world_size, rank=rank, drop_last=True, shuffle=True)
    )

    valpaths = sorted(pathlib.Path("cache/merge_tensordataset/multitask_tensors/tst").glob("*.pt"))
    valpaths = valpaths[rank:] + valpaths[:rank]
    valds = InMemorySequenceShiftDataset.from_cache_or_create(
        valpaths, tokenizer, nprops=10, assay_filter=tox21_props,
        cache_path= outdir / "inmemory_dataset_cache" / "val_dataset_tox21.pt")

    valdl = torch.utils.data.DataLoader(
        valds, batch_size=batch_size, shuffle=False, num_workers=val_workers,
        pin_memory=True, persistent_workers=True, prefetch_factor=100,
        sampler=torch.utils.data.distributed.DistributedSampler(valds, num_replicas=world_size, rank=rank, drop_last=True, shuffle=True)
    )
    
    logging.info("creating trainer")
    num_batches_to_accumulate = 8
    warmup_steps = 2000
    first_cycle_length = 1000 
    trainer = Trainer(model, rank, tokenizer, trndl, batch_size=batch_size, 
        scheduler_warmup_steps=warmup_steps, 
        scheduler_max_steps=first_cycle_length, 
        scheduler_min_lr=1e-6, 
        scheduler_max_lr=1e-3, 
        effective_accum_batch_size=batch_size * world_size * num_batches_to_accumulate,
        max_steps=1_000_000, eval_samples=2_000, first_eval=1000, eval_every=100)
    
    trainer.set_validation_dataloader(valdl)
    trainer.set_mask_percent(0.1)
    trainer.set_model_savepath(modeldir / "moe")
    trainer.set_metrics_file(metricsdir / "multitask_loss.tsv", overwrite=True)

    if rank == 0:
        logging.info(f"trnds samples: {len(trnds)}, valds samples: {len(valds)}")
        logging.info(f"workers train: {train_workers}, val: {val_workers}")
        logging.info(f"{len(trndl)} train batches")
        logging.info(f"{len(valdl)} validation batches")
        logging.info(f"{num_params/1e6:.2f} million parameters")
        logging.info(f"Gradient accumulation: {trainer.gradient_accumulation_steps}")
        trainable_params_count = len([p for p in model.parameters() if p.requires_grad])
        logging.info(f"Total trainable parameters: {trainable_params_count}")


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

    torch.cuda.set_device(rank)
    dist.init_process_group(
        backend="nccl",
        init_method="env://",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(minutes=30),
    )
    
    print(f"RANK {rank} starting with WORLD_SIZE {world_size}")
    main(rank, world_size)

    dist.destroy_process_group()
