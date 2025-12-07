# CMD
"""
mkdir -p cache/generate_evaluations_holdout_context/logs
PYTHONPATH=./ torchrun --nproc_per_node=2 \
    code/5_0_0_3_generate_evaluations_holdout_context.py \
    2>&1 | tee cache/generate_evaluations_holdout_context/logs/run.log
"""
"""
Generate context-aware evaluations from holdout tensor files.

This extends 5_0_0_2 by prepending known properties for each compound as context.
Properties are filtered by mutual information with the target property, using
output from the token_information stage.

Key features:
- Looks up known properties for each compound from SQLite
- Filters context properties by MI with target property
- Prepends up to MAX_CONTEXT_PROPERTIES to property/value sequences
- Evaluates each sample with its contextual information
"""
import os
import sqlite3
import logging
import pathlib
from dataclasses import dataclass

import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from tqdm import tqdm
import torch.distributed as dist

from cvae.tokenizer import SelfiesPropertyValTokenizer
import cvae.models.multitask_encoder as mte


# =============================================================================
# Configuration
# =============================================================================

@dataclass
class Config:
    """Evaluation configuration."""
    splits: tuple = (0, 1, 2, 3, 4)  # All 5 bootstrap splits
    batch_size: int = 5120
    max_context_properties: int = 20
    min_mi_threshold: float = 0.07  # Minimum MI to include property as context
    output_dir: str = 'cache/generate_evaluations_holdout_context'
    db_path: str = 'brick/cvae.sqlite'
    mi_path: str = 'cache/token_information/pairwise_mi.parquet'
    tensor_base: str = 'cache/build_tensordataset/bootstrap'
    model_base: str = 'cache/train_multitask_transformer_parallel/logs'


# Context windows for evaluation - each sample gets evaluated at these fixed sizes
# (min_context, exact_context, bucket_label)
CONTEXT_WINDOWS = [
    (1, 5, "1-5"),      # Use exactly 5 context items (if >= 5 available)
    (6, 10, "6-10"),    # Use exactly 10 context items (if >= 10 available)
    (11, 20, "11-20"),  # Use exactly 20 context items (if >= 20 available)
]


CONFIG = Config()


# =============================================================================
# Data Loading - Tidyverse-style Pipeline Functions
# =============================================================================

def load_pairwise_mi(path: str) -> pd.DataFrame:
    """Load and prepare pairwise mutual information data."""
    return (
        pd.read_parquet(path)
        .assign(
            prop1=lambda df: df['prop1'].astype(int),
            prop2=lambda df: df['prop2'].astype(int),
        )
    )


def build_mi_lookup(mi_df: pd.DataFrame, min_threshold: float = 0.0) -> dict:
    """
    Build a lookup: target_property -> [(context_property, mi_score), ...].

    Properties are sorted by MI descending for easy top-k selection.
    """
    # Filter by threshold
    filtered = mi_df[mi_df['mi'] > min_threshold].copy()

    # Create bidirectional mapping (MI is symmetric)
    forward = filtered[['prop1', 'prop2', 'mi']].rename(
        columns={'prop1': 'target', 'prop2': 'context'}
    )
    backward = filtered[['prop2', 'prop1', 'mi']].rename(
        columns={'prop2': 'target', 'prop1': 'context'}
    )
    combined = pd.concat([forward, backward], ignore_index=True)

    # Group by target and sort by MI
    lookup = {}
    for target, group in combined.groupby('target'):
        sorted_pairs = (
            group
            .sort_values('mi', ascending=False)
            .drop_duplicates('context')
            [['context', 'mi']]
            .values.tolist()
        )
        lookup[int(target)] = [(int(c), float(m)) for c, m in sorted_pairs]

    return lookup


def load_compound_properties(db_path: str, relevant_properties: set = None) -> pd.DataFrame:
    """
    Load compound-property-value mappings from SQLite, keyed by selfies_tokens.

    Args:
        db_path: Path to SQLite database
        relevant_properties: If provided, only load these property tokens (for efficiency)
    """
    conn = sqlite3.connect(db_path)

    if relevant_properties and len(relevant_properties) < 10000:
        # Filter to relevant properties for efficiency
        props_str = ','.join(str(p) for p in relevant_properties)
        query = f"""
            SELECT selfies_tokens, property_token, value
            FROM activity
            WHERE selfies_tokens IS NOT NULL
              AND property_token IN ({props_str})
        """
    else:
        query = """
            SELECT selfies_tokens, property_token, value
            FROM activity
            WHERE selfies_tokens IS NOT NULL
              AND property_token IS NOT NULL
        """

    logging.info("Loading compound properties from SQLite...")
    df = pd.read_sql(query, conn)
    conn.close()

    logging.info(f"Loaded {len(df):,} activity records")
    return df.assign(property_token=lambda x: x['property_token'].astype(int))


def build_compound_lookup(df: pd.DataFrame) -> dict:
    """
    Build lookup: selfies_tokens -> {property_token: value, ...}.
    """
    lookup = {}
    for selfies_tokens, group in df.groupby('selfies_tokens'):
        lookup[selfies_tokens] = dict(zip(group['property_token'], group['value']))
    return lookup


def load_holdout_selfies_tokens(db_path: str, split_id: int) -> set:
    """Load selfies_tokens strings in the holdout set for a split."""
    conn = sqlite3.connect(db_path)
    query = f"SELECT DISTINCT selfies_tokens FROM holdout_samples WHERE split_id = {split_id}"
    df = pd.read_sql(query, conn)
    conn.close()
    return set(df['selfies_tokens'].tolist())


# =============================================================================
# Context Selection Logic
# =============================================================================

def select_context_properties(
    target_property: int,
    available_properties: dict,
    mi_lookup: dict,
    max_context: int,
    exclude_target: bool = True,
) -> list:
    """
    Select context properties for a target, ranked by mutual information.

    Args:
        target_property: The property we're predicting
        available_properties: {property_token: value} for this compound
        mi_lookup: target -> [(context, mi), ...] sorted by MI descending
        max_context: Maximum number of context properties
        exclude_target: Whether to exclude the target from context

    Returns:
        List of (property_token, value) tuples, MI-ranked
    """
    if target_property not in mi_lookup:
        return []

    context = []
    for context_prop, _ in mi_lookup[target_property]:
        if exclude_target and context_prop == target_property:
            continue
        if context_prop in available_properties:
            context.append((context_prop, available_properties[context_prop]))
            if len(context) >= max_context:
                break

    return context


# =============================================================================
# Dataset
# =============================================================================

class HoldoutContextDataset(Dataset):
    """
    Dataset that loads holdout samples and enriches with context properties.

    Creates multiple samples per (selfies, property) at fixed context windows:
    - "1-5": exactly 5 context items (if available)
    - "6-10": exactly 10 context items (if available)
    - "11-20": exactly 20 context items (if available)
    """

    def __init__(
        self,
        split_id: int,
        tokenizer,
        compound_lookup: dict,
        mi_lookup: dict,
        holdout_selfies_tokens: set,
        config: Config,
        rank: int = 0,
    ):
        self.tokenizer = tokenizer
        self.config = config
        self.holdout_selfies_tokens = holdout_selfies_tokens
        self.samples = []

        # Track stats per context window
        window_counts = {label: 0 for _, _, label in CONTEXT_WINDOWS}

        # Load tensor files
        test_dir = pathlib.Path(f"{config.tensor_base}/split_{split_id}/test")
        tensor_files = sorted(test_dir.glob("*.pt"))

        logging.info(f"Rank {rank}: Loading {len(tensor_files)} tensor files...")

        for tensor_file in tqdm(tensor_files, desc=f"Split {split_id}", disable=rank != 0):
            data = torch.load(tensor_file, map_location="cpu", weights_only=True)

            selfies_tensors = data["selfies"]
            properties_tensors = data["properties"]
            values_tensors = data["values"]

            for selfies_t, props, vals in zip(selfies_tensors, properties_tensors, values_tensors):
                # Convert selfies tensor to space-joined token string
                selfies_tokens = ' '.join(str(x) for x in selfies_t.tolist())

                if selfies_tokens not in compound_lookup:
                    continue

                available_props = compound_lookup[selfies_tokens]

                # Get valid property-value pairs from tensor
                valid_mask = (props != -1) & (vals != -1)
                valid_props = props[valid_mask].tolist()
                valid_vals = vals[valid_mask].tolist()

                # Create samples for each property at each context window
                for target_prop, target_val in zip(valid_props, valid_vals):
                    # Get all available context (up to 20, ranked by MI)
                    all_context = select_context_properties(
                        target_property=target_prop,
                        available_properties=available_props,
                        mi_lookup=mi_lookup,
                        max_context=config.max_context_properties,
                        exclude_target=True,
                    )

                    n_available = len(all_context)

                    # Create a sample for each context window this target qualifies for
                    for min_ctx, exact_ctx, bucket_label in CONTEXT_WINDOWS:
                        if n_available >= min_ctx:
                            # Use exactly exact_ctx items, or all available if less
                            n_to_use = min(exact_ctx, n_available)
                            context_subset = all_context[:n_to_use]

                            self.samples.append({
                                'selfies': selfies_t,
                                'selfies_tokens': selfies_tokens,
                                'target_property': target_prop,
                                'target_value': target_val,
                                'context': context_subset,
                                'context_bucket': bucket_label,
                            })
                            window_counts[bucket_label] += 1

        logging.info(f"Rank {rank}: Created {len(self.samples)} context-enriched samples")

        # Log context window statistics
        for label, count in window_counts.items():
            logging.info(f"Rank {rank}: Context window '{label}': {count:,} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


# =============================================================================
# Collation
# =============================================================================

def collate_with_context(batch: list, max_seq_len: int = 21) -> dict:
    """
    Collate batch with context prepended to property/value sequences.

    Property sequence: [context_prop_1, ..., context_prop_n, target_prop]
    Value sequence:    [context_val_1, ..., context_val_n, target_val]
    """
    batch_size = len(batch)

    # Determine actual sequence lengths
    seq_lens = [len(item['context']) + 1 for item in batch]
    max_len = min(max(seq_lens), max_seq_len)

    # Initialize tensors - use 0 for padding (valid embedding index)
    # The mask will ensure padding positions are ignored
    selfies = torch.stack([item['selfies'] for item in batch])
    properties = torch.zeros((batch_size, max_len), dtype=torch.long)
    values = torch.zeros((batch_size, max_len), dtype=torch.long)
    mask = torch.zeros((batch_size, max_len), dtype=torch.bool)

    # Fill sequences: context first, then target
    for i, item in enumerate(batch):
        context = item['context']

        # Context properties and values
        for j, (prop, val) in enumerate(context[:max_len - 1]):
            properties[i, j] = prop
            values[i, j] = val
            mask[i, j] = True

        # Target property and value (last position)
        target_pos = min(len(context), max_len - 1)
        properties[i, target_pos] = item['target_property']
        values[i, target_pos] = item['target_value']
        mask[i, target_pos] = True

    return {
        'selfies': selfies,
        'properties': properties,
        'values': values,
        'mask': mask,
        'true_values': torch.tensor([item['target_value'] for item in batch], dtype=torch.long),
        'property_tokens': torch.tensor([item['target_property'] for item in batch], dtype=torch.long),
        'n_context': torch.tensor([len(item['context']) for item in batch], dtype=torch.long),
        'selfies_tokens': [item['selfies_tokens'] for item in batch],
        'context_bucket': [item['context_bucket'] for item in batch],
    }


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
) -> list:
    """Evaluate one split and return results."""
    model.eval()
    results = []

    for batch in tqdm(dataloader, desc=f"Split {split_id}", disable=rank != 0):
        # Move to device
        selfies = batch['selfies'].to(device)
        properties = batch['properties'].to(device)
        values = batch['values'].to(device)
        mask = batch['mask'].to(device)

        # Forward pass - model predicts at each position
        logits = model(selfies, properties, values, mask)

        # Get prediction for the last valid position (target)
        batch_size = selfies.shape[0]
        last_positions = mask.sum(dim=1) - 1  # Index of last True in each row

        # Extract logits at target position
        target_logits = logits[torch.arange(batch_size), last_positions]  # [batch, 2]
        probs = torch.softmax(target_logits, dim=-1)[:, 1].cpu().numpy()

        # Collect results
        for i in range(batch_size):
            results.append({
                'selfies_tokens': batch['selfies_tokens'][i],
                'property_token': int(batch['property_tokens'][i]),
                'true_value': int(batch['true_values'][i]),
                'prob_of_1': float(probs[i]),
                'n_context': int(batch['n_context'][i]),
                'context_bucket': batch['context_bucket'][i],
                'split': split_id,
            })

    return results


# =============================================================================
# Distributed Setup
# =============================================================================

def setup_distributed() -> tuple:
    """Setup distributed training, return (rank, world_size, device)."""
    if 'RANK' in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ["LOCAL_RANK"])
        device = torch.device(f'cuda:{local_rank}')
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
    else:
        rank = 0
        world_size = 0
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    return rank, world_size, device


def setup_logging(output_dir: pathlib.Path, rank: int):
    """Configure logging to both console and file."""
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"rank_{rank}.log"

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_file, mode='w'),
        ]
    )


# =============================================================================
# Main
# =============================================================================

def main():
    config = CONFIG

    # Setup distributed first to get rank
    rank, world_size, device = setup_distributed()

    # Create output directory and setup logging
    output_dir = pathlib.Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir, rank)

    # Load shared resources (once per rank)
    logging.info(f"Rank {rank}: Loading tokenizer...")
    tokenizer = SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

    logging.info(f"Rank {rank}: Loading mutual information data...")
    mi_df = load_pairwise_mi(config.mi_path)
    mi_lookup = build_mi_lookup(mi_df, config.min_mi_threshold)
    logging.info(f"Rank {rank}: MI lookup has {len(mi_lookup)} target properties")

    # Extract all properties that could be used as context
    relevant_properties = set()
    for target, context_list in mi_lookup.items():
        relevant_properties.add(target)
        relevant_properties.update(ctx_prop for ctx_prop, _ in context_list)
    logging.info(f"Rank {rank}: {len(relevant_properties)} relevant properties for context")

    logging.info(f"Rank {rank}: Loading compound properties from SQLite...")
    compound_df = load_compound_properties(config.db_path, relevant_properties)
    compound_lookup = build_compound_lookup(compound_df)
    logging.info(f"Rank {rank}: Compound lookup has {len(compound_lookup)} compounds")

    # Process each split
    total_results = 0
    for split_id in config.splits:
        logging.info(f"Rank {rank}: Processing split {split_id}")

        # Load holdout selfies_tokens for this split (for logging/verification)
        holdout_selfies_tokens = load_holdout_selfies_tokens(config.db_path, split_id)
        logging.info(f"Rank {rank}: Holdout has {len(holdout_selfies_tokens)} compounds for split {split_id}")

        # Load model for this split
        model_path = f'{config.model_base}/split_{split_id}/models/me_roundrobin_property_dropout_V3/best_loss'
        model = mte.MultitaskEncoder.load(model_path)
        model.to(device)

        # Create dataset
        dataset = HoldoutContextDataset(
            split_id=split_id,
            tokenizer=tokenizer,
            compound_lookup=compound_lookup,
            mi_lookup=mi_lookup,
            holdout_selfies_tokens=holdout_selfies_tokens,
            config=config,
            rank=rank,
        )

        # Create dataloader
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank) if world_size > 0 else None
        dataloader = DataLoader(
            dataset,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=4,
            pin_memory=True,
            collate_fn=collate_with_context,
        )

        # Evaluate
        results = evaluate_split(model, dataloader, device, rank, split_id)

        # Save results
        if results:
            df = pd.DataFrame(results)
            split_file = output_dir / "evaluations.parquet" / f"split_{split_id}_rank_{rank}.parquet"
            split_file.parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(split_file, index=False)
            total_results += len(results)
            logging.info(f"Rank {rank}: Saved {len(results)} results for split {split_id}")

    # Cleanup
    if world_size > 0:
        dist.barrier()
        dist.destroy_process_group()

    logging.info(f"Rank {rank}: Completed with {total_results} total results")
    if rank == 0:
        logging.info(f"All results saved to {output_dir}/")


if __name__ == "__main__":
    main()