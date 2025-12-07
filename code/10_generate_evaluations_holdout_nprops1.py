# CMD
"""
PYTHONPATH=./ torchrun --nproc_per_node=2 \
    code/5_0_0_2_generate_evaluations_holdout_nprops1.py
"""
"""
Generate nprops=1 evaluations from holdout tensor files.

Evaluates each (selfies, property, value) pair independently - no context.
This is the baseline for comparing against context-aware methods.

Outputs:
    cache/generate_evaluations_holdout_nprops1/
    ├── evaluations.parquet/
    │   ├── split_0_rank_0.parquet
    │   └── ...
    ├── out.log
    └── err.log
"""
import os
import sys
import logging
import pathlib
import time

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm
import torch.distributed as dist

from cvae.tokenizer import SelfiesPropertyValTokenizer
from cvae.datasets import HoldoutDataset
from cvae.datasets.holdout_dataset import collate_nprops1
import cvae.models.multitask_encoder as mte


# =============================================================================
# Configuration
# =============================================================================

SPLITS = (0, 1, 2, 3, 4)
BATCH_SIZE = 5120
OUTPUT_DIR = 'cache/generate_evaluations_holdout_nprops1'
TENSOR_BASE = 'cache/build_tensordataset/bootstrap'
MODEL_BASE = 'cache/train_multitask_transformer_parallel/logs'


# =============================================================================
# Logging Setup
# =============================================================================

def setup_logging(output_dir: pathlib.Path, rank: int) -> logging.Logger:
    """Configure logging with file handlers."""
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Console handler
    console_handler = logging.StreamHandler(sys.stderr)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # File handlers (rank 0 only)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)

        out_handler = logging.FileHandler(output_dir / 'out.log', mode='w')
        out_handler.setLevel(logging.INFO)
        out_handler.setFormatter(formatter)
        logger.addHandler(out_handler)

        err_handler = logging.FileHandler(output_dir / 'err.log', mode='w')
        err_handler.setLevel(logging.WARNING)
        err_handler.setFormatter(formatter)
        logger.addHandler(err_handler)

    return logger


# =============================================================================
# Distributed Setup
# =============================================================================

def setup_distributed() -> tuple:
    """Setup distributed training."""
    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        return rank, world_size, device
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        return 0, 1, device


def cleanup_distributed(world_size: int):
    """Clean up distributed process group."""
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


# =============================================================================
# Evaluation
# =============================================================================

@torch.no_grad()
def evaluate_split(
    model,
    dataloader: DataLoader,
    device: torch.device,
    rank: int,
    split_id: int,
    total_splits: int,
) -> list:
    """Evaluate one split and return results."""
    model.eval()
    results = []

    pbar = tqdm(
        dataloader,
        desc=f"Eval split {split_id}/{total_splits-1}",
        disable=rank != 0,
        unit="batch",
        dynamic_ncols=True,
    )

    for batch in pbar:
        # Move to device
        selfies = batch['selfies'].to(device)
        properties = batch['properties'].to(device)
        values = batch['values'].to(device)
        mask = batch['mask'].to(device)

        # Forward pass
        logits = model(selfies, properties, values, mask)  # [batch, 1, 2]
        probs = torch.softmax(logits, dim=-1)[:, 0, 1].cpu().numpy()

        # Collect results
        for i in range(len(batch['property_tokens'])):
            results.append({
                'selfies_tokens': batch['selfies_tokens'][i],
                'property_token': int(batch['property_tokens'][i]),
                'true_value': int(batch['true_values'][i]),
                'prob_of_1': float(probs[i]),
                'split': split_id,
            })

        pbar.set_postfix({'samples': f'{len(results):,}'}, refresh=False)

    return results


# =============================================================================
# Main
# =============================================================================

def main():
    start_time = time.time()

    # Setup distributed
    rank, world_size, device = setup_distributed()

    # Setup logging
    output_dir = pathlib.Path(OUTPUT_DIR)
    setup_logging(output_dir, rank)

    logging.info(f"[Rank {rank}] Starting nprops=1 evaluation")
    logging.info(f"[Rank {rank}] Device: {device}, World size: {world_size}")

    if rank == 0:
        logging.info("=" * 70)
        logging.info("NPROPS=1 EVALUATION (No Context Baseline)")
        logging.info("=" * 70)
        logging.info(f"Output directory: {output_dir}")
        logging.info(f"Splits: {SPLITS}")
        logging.info(f"Batch size: {BATCH_SIZE}")
        logging.info(f"World size (GPUs): {world_size}")
        logging.info("=" * 70)

    # Load tokenizer
    logging.info(f"[Rank {rank}] Loading tokenizer...")
    tokenizer = SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

    # Process each split
    total_results = 0
    total_splits = len(SPLITS)

    for split_idx, split_id in enumerate(SPLITS):
        split_start_time = time.time()

        if rank == 0:
            logging.info("")
            logging.info("=" * 70)
            logging.info(f"SPLIT {split_id} ({split_idx + 1}/{total_splits})")
            logging.info("=" * 70)

        # Load model for this split
        model_path = f'{MODEL_BASE}/split_{split_id}/models/me_roundrobin_property_dropout_V3/best_loss'
        logging.info(f"[Rank {rank}] Loading model from {model_path}")
        model = mte.MultitaskEncoder.load(model_path)
        model.to(device)

        # Create dataset using shared HoldoutDataset
        dataset = HoldoutDataset(
            split_id=split_id,
            tensor_base=TENSOR_BASE,
            rank=rank,
            include_compound_properties=False,  # Not needed for nprops=1
        )

        # Create dataloader
        if world_size > 1:
            sampler = DistributedSampler(
                dataset, num_replicas=world_size, rank=rank, shuffle=False
            )
        else:
            sampler = None

        dataloader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            sampler=sampler,
            shuffle=False,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_nprops1,
        )

        if rank == 0:
            logging.info(f"Evaluating {len(dataset):,} samples ({len(dataloader)} batches)...")

        # Evaluate
        results = evaluate_split(model, dataloader, device, rank, split_id, total_splits)

        # Save results
        if results:
            df = pd.DataFrame(results)
            split_file = output_dir / "evaluations.parquet" / f"split_{split_id}_rank_{rank}.parquet"
            split_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(split_file, index=False)
            total_results += len(results)

            split_elapsed = time.time() - split_start_time
            samples_per_sec = len(results) / split_elapsed if split_elapsed > 0 else 0

            logging.info(f"[Rank {rank}] Split {split_id} complete:")
            logging.info(f"  Samples: {len(results):,}")
            logging.info(f"  Elapsed time: {split_elapsed/60:.1f} min ({samples_per_sec:.0f} samples/sec)")
            logging.info(f"  Saved to: {split_file}")

        # Sync between splits
        if world_size > 1:
            dist.barrier()

    # Cleanup
    cleanup_distributed(world_size)

    # Final summary
    total_elapsed = time.time() - start_time

    if rank == 0:
        logging.info("")
        logging.info("=" * 70)
        logging.info("NPROPS=1 EVALUATION COMPLETE")
        logging.info("=" * 70)
        logging.info(f"Total samples: {total_results:,}")
        logging.info(f"Splits processed: {total_splits}")
        logging.info(f"Total elapsed time: {total_elapsed/60:.1f} min")
        logging.info(f"Output directory: {output_dir}/")
        logging.info("=" * 70)
    else:
        logging.info(f"[Rank {rank}] Completed with {total_results:,} results")


if __name__ == "__main__":
    main()
