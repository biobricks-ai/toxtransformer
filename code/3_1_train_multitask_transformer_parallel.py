#!/usr/bin/env python
# -*- coding: utf-8 -*-
# PYTHONPATH=./ CUDA_LAUNCH_BLOCKING=1 torchrun --standalone --nproc-per-node=8 --master-port=29500 code/3_1_train_multitask_transformer_parallel.py 2> cache/train_multitask_transformer_parallel/logs/err.log

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
    if rank == 0:
        logdir = pathlib.Path("cache/train_multitask_transformer_parallel/logs")
        logdir.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(filename=logdir / "train_multitask_transformer_parallel.log", level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
        logging.info("Logging initialized.")

def main(rank, world_size):
    setup(rank, world_size)
    init_logging(rank)

    outdir = pathlib.Path("cache/train_multitask_transformer_parallel")
    modeldir = outdir / "models"
    metricsdir = outdir / "metrics"

    logging.info(f"Rank {rank} starting setup.")
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

    # Epoch: 0, Step: 6700, Train Loss (last cycle): 0.3774, Eval Loss: 0.3448, BAC: 0.7082, AUC: 0.8653, LR: 0.000020
    # model = me.MoE(tokenizer, num_experts=16, k=4, hdim=256, dim_feedforward=1024, nhead=4, balance_loss_weight=0.1, expert_layers=6)
    # batch_size = 100

    # 2025-04-27 19:51:22 - INFO - Epoch: 0, Step: 1700, Train Loss (last cycle): 2.1533, Eval Loss: 0.4625, BAC: 0.5000, AUC: 0.7146, LR: 0.000078
    # model = me.MoE(tokenizer, num_experts=16, k=4, hdim=512, dim_feedforward=2048, nhead=4, balance_loss_weight=0.1, expert_layers=6)
    # batch_size = 25

    # 2025-04-28 08:57:34 - INFO - Rank 0: Evaluation complete. Loss: 0.5168, AUC: 0.7997, BAC: 0.6644
    # model = me.MoE(tokenizer, num_experts=18, k=4, hdim=512, dim_feedforward=2048, nhead=4, balance_loss_weight=0.1, expert_layers=6)
    # batch_size = 25

    # 2025-04-28 20:12:00 - INFO - Epoch: 0, Step: 6000, Train Loss (last cycle): 0.4948, Eval Loss: 0.4219, BAC: 0.7163, AUC: 0.8700, LR: 0.000100
    # 2025-05-06 13:02:38 - INFO - Epoch: 8, Step: 37000, Train Loss (last cycle): 0.9009, Eval Loss: 0.8615, BAC: 0.7482, AUC: 0.9135, LR: 0.000210
    # model = me.MoE(tokenizer, num_experts=24, k=4, hdim=512, dim_feedforward=2048, nhead=4, balance_loss_weight=0.1, diversity_loss_weight=1e-4, expert_layers=6)

    # model = me.MoE(tokenizer, num_experts=12, k=4, hdim=512, dim_feedforward=1024, nhead=4, balance_loss_weight=0.1, diversity_loss_weight=1e-4, expert_layers=6)
    # Epoch: 30, Step: 310000, Train Loss (last cycle): 0.7030, Eval Loss: 0.7256, BAC: 0.7734, AUC: 0.9643, LR: 0.000001
    # model = me.MoE.load("cache/train_multitask_transformer_parallel/models/moe")
    # model = me.MoE(tokenizer, num_experts=12, k=4, hdim=512, dim_feedforward=1024, nhead=4, balance_loss_weight=0.1, diversity_loss_weight=1e-4, expert_layers=6)
    model = me.MoE.load("cache/train_multitask_transformer_parallel/models/moe")
    batch_size = 42

    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    cpus_per_rank = max(2, (os.cpu_count() or 8) // world_size)
    train_workers = max(2, min(4, cpus_per_rank))
    val_workers = max(1, min(2, cpus_per_rank))
    nprops = 1

    # trnds = sd.SamplingDataset("cache/build_tensordataset/multitask_tensors/trn", tokenizer, nprops=50, sample_pool=int(5e4), recent_prop_cap=4000, recent_idx_cap=int(1e6))
    # valds = sd.SamplingDataset("cache/build_tensordataset/multitask_tensors/tst", tokenizer, nprops=50, sample_pool=int(5e4), recent_prop_cap=4000, recent_idx_cap=int(1e6))
    bp = pd.read_parquet("cache/get_benchmark_properties/benchmark_properties.parquet")
    target_props = list(set(bp['property_token'].tolist()))
    
    trnds = mt.SequenceShiftDataset("cache/build_tensordataset/multitask_tensors/trn", tokenizer, nprops=100)
    valds = mt.SequenceShiftDataset("cache/build_tensordataset/multitask_tensors/tst", tokenizer, nprops=100, assay_filter=target_props)

   # Create rank-aware datasets using setup_distributed
    # trnds = mt.RotatingModuloSequenceShiftDataset.setup_distributed(
    #     path="cache/build_tensordataset/multitask_tensors/trn", 
    #     tokenizer=tokenizer, 
    #     nprops=100,
    #     local_rank=rank,
    #     world_size=world_size
    # )
    
    # valds = mt.RotatingModuloSequenceShiftDataset.setup_distributed(
    #     path="cache/build_tensordataset/multitask_tensors/tst", 
    #     tokenizer=tokenizer, 
    #     nprops=100,
    #     local_rank=rank,
    #     world_size=world_size
    # )

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

    trainer = Trainer(model, rank, tokenizer, trndl, batch_size=batch_size, 
        scheduler_warmup_steps=10_000, scheduler_max_steps=300_000, scheduler_min_lr=2e-5, scheduler_max_lr=5e-4,
        max_steps=1_000_000)
    
    trainer.set_validation_dataloader(valdl)
    trainer.set_mask_percent(0.1)
    trainer.set_model_savepath(modeldir / "moe")
    trainer.set_metrics_file(metricsdir / "multitask_loss.tsv", overwrite=True)

    if rank == 0:
        logging.info(f"trnds samples: {len(trnds)}, valds samples: {len(valds)}")
        logging.info(f"workers train: {train_workers}, val: {val_workers}")
        trainer.log(f"{len(trndl)} train batches")
        trainer.log(f"{len(valdl)} validation batches")
        trainer.log(f"{num_params/1e6:.2f} million parameters")
        trainer.log(f"Gradient accumulation: {trainer.gradient_accumulation_steps}")

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
