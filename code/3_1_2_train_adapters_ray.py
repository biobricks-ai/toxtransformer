import os
import sys
import shutil
import pathlib
import logging
import datetime
import time

import torch
import torch.utils.data
import torch.multiprocessing as mp
import torch.distributed as dist
import pandas as pd

import ray

import cvae.tokenizer
import cvae.utils
import cvae.models.multitask_transformer as mt
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
os.environ['TORCH_NCCL_ASYNC_ERROR_HANDLING'] = '1'
os.environ['NCCL_IB_DISABLE'] = '0'
os.environ['NCCL_P2P_DISABLE'] = '0'

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
mp.set_start_method("fork", force=True)

@ray.remote
class DatasetHolder:
    def __init__(self):
        tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')
        bp = pd.read_parquet("cache/get_benchmark_properties/benchmark_properties.parquet")
        tox21_props = bp[bp['source'].isin(['Tox21'])]['property_token'].unique().tolist()

        trnpaths = sorted(pathlib.Path("cache/merge_tensordataset/multitask_tensors/trn").glob("*.pt"))
        valpaths = sorted(pathlib.Path("cache/merge_tensordataset/multitask_tensors/tst").glob("*.pt"))

        self.trnds = InMemorySequenceShiftDataset.from_cache_or_create(
            trnpaths, tokenizer, nprops=10, assay_filter=tox21_props,
            cache_path=pathlib.Path("cache/train_adapters/inmemory_dataset_cache") / "trn_dataset_tox21.pt")

        self.valds = InMemorySequenceShiftDataset.from_cache_or_create(
            valpaths, tokenizer, nprops=10, assay_filter=tox21_props,
            cache_path=pathlib.Path("cache/train_adapters/inmemory_dataset_cache") / "val_dataset_tox21.pt")

    def get_dataset_refs(self):
        return ray.put(self.trnds), ray.put(self.valds)

def setup_ray_and_datasets(rank, world_size):
    if not ray.is_initialized():
        ray.init(address="auto", ignore_reinit_error=True)

    actor_name = "dataset_holder"

    if rank == 0:
        try:
            ray.get_actor(actor_name)
        except ValueError:
            DatasetHolder.options(name=actor_name, lifetime="detached").remote()

    while True:
        try:
            ray.get_actor(actor_name)
            break
        except ValueError:
            time.sleep(1)

    actor = ray.get_actor(actor_name)
    trn_ref, val_ref = ray.get(actor.get_dataset_refs.remote())
    return trn_ref, val_ref

def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=30))

def cleanup():
    if dist.is_initialized():
        dist.destroy_process_group()

def init_logging(rank):
    if rank == 0:
        logdir = pathlib.Path("cache/train_adapters/logs")
        logdir.mkdir(exist_ok=True, parents=True)
        logging.basicConfig(filename=logdir / "train_adapters.log", level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s', filemode='w')
        logging.info("Logging initialized.")

def main(rank, world_size):
    setup(rank, world_size)
    torch.cuda.set_device(rank)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

    init_logging(rank)

    outdir = pathlib.Path("cache/train_adapters")
    modeldir = outdir / "models"
    modeldir.mkdir(exist_ok=True, parents=True)
    metricsdir = outdir / "metrics"
    metricsdir.mkdir(exist_ok=True, parents=True)

    logging.info(f"Rank {rank} starting setup.")
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

    cpus_per_rank = max(2, os.cpu_count() // world_size)
    train_workers = max(2, min(32, cpus_per_rank))
    val_workers = max(1, min(20, cpus_per_rank))

    bp = pd.read_parquet("cache/get_benchmark_properties/benchmark_properties.parquet")
    benchmark_properties = list(bp['property_token'].unique())
    tox21_props = list(bp[bp['source'].isin(['Tox21'])]['property_token'].unique().tolist())

    model : me.MoE = me.MoE.load("cache/train_multitask_transformer_parallel/models/step_16000")
    model.save(modeldir / "moe")
    model.freeze_shared_embedding()
    # model.freeze_main_experts()

    adapter_props = tox21_props
    for prop in adapter_props:
        _ = model.add_boosting_expert(prop, expert_layers=4, expert_nhead=4, expert_dim_feedforward=256, expert_dropout_rate=0.1)
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    batch_size = 32

    trn_ref, val_ref = setup_ray_and_datasets(rank, world_size)
    trnds = ray.get(trn_ref)
    valds = ray.get(val_ref)

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

    trainer = Trainer(model, rank, tokenizer, trndl, batch_size=batch_size, 
        scheduler_warmup_steps=10_000, scheduler_max_steps=300_000, 
        scheduler_min_lr=2e-5, scheduler_max_lr=1e-4, effective_accum_batch_size=2048 * 8,
        max_steps=1_000_000, eval_samples=20000, first_eval=1000, eval_every=100)

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

    trainer.start()
    cleanup()

    if rank == 0:
        logging.info("Copying final model from modeldir to brick/moe...")
        if os.path.exists("brick/moe"):
            shutil.rmtree("brick/moe", ignore_errors=True)
        shutil.copytree(modeldir / "moe", "brick/moe")

if __name__ == "__main__":
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    print(f"RANK {rank} starting with WORLD_SIZE {world_size}")
    main(rank, world_size)
