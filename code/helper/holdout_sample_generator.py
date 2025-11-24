"""
Generate evaluation samples for holdout sets using the activity table with selfies column.

This module queries holdout_samples and activity tables (both now have selfies columns)
to generate samples with real property values for evaluation.
"""

import torch
import sqlite3
import pandas as pd
from typing import Tuple, Optional, Generator, Dict
import random


def generate_holdout_samples(
    selfies: str,
    property_token: int,
    tokenizer,
    nprops: int,
    num_samples: int = 50,
    db_path: str = "brick/cvae.sqlite",
    pad_idx: int = 0,
    device: str = 'cpu',
    seed: Optional[int] = None,
    properties_cache: Optional[dict] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """
    Generate samples for a holdout (selfies, property_token) pair using real property values.

    Args:
        selfies: SELFIES string
        property_token: Target property to predict (placed at position nprops-1)
        tokenizer: SelfiesPropertyValTokenizer instance
        nprops: Number of properties per sample
        num_samples: Number of distinct samples to generate
        db_path: Path to SQLite database
        pad_idx: Padding index (default: 0)
        device: Device to place tensors on
        seed: Random seed for reproducibility
        properties_cache: Optional dict mapping selfies -> DataFrame of properties
                         to avoid repeated database queries

    Returns:
        (selfies_batch, properties, values, mask, true_value)
        - selfies_batch: [num_samples, seq_len]
        - properties: [num_samples, nprops]
        - values: [num_samples, nprops]
        - mask: [num_samples, nprops]
        - true_value: Ground truth for target property
    """
    if seed is not None:
        random.seed(seed)
        torch.manual_seed(seed)

    device_obj = torch.device(device)

    # Use cache if available, otherwise query database
    if properties_cache is not None and selfies in properties_cache:
        all_props = properties_cache[selfies]
    else:
        conn = sqlite3.connect(db_path)
        # Get all properties for this molecule (using selfies join)
        query = """
        SELECT DISTINCT property_token, value
        FROM activity
        WHERE selfies = ?
        """
        all_props = pd.read_sql_query(query, conn, params=(selfies,))
        conn.close()

        # Cache for future use
        if properties_cache is not None:
            properties_cache[selfies] = all_props

    if len(all_props) == 0:
        raise ValueError(f"No properties found for {selfies}")

    # Get target property value
    target = all_props[all_props['property_token'] == property_token]
    if len(target) == 0:
        raise ValueError(f"Property {property_token} not found for {selfies}")

    true_value = int(target['value'].iloc[0])

    # Get context properties (excluding target)
    context = all_props[all_props['property_token'] != property_token]
    if len(context) < nprops - 1:
        raise ValueError(f"Need {nprops-1} context properties, have {len(context)}")

    # Encode SELFIES with padding/truncation to fixed length (120 tokens)
    # This matches how SELFIES were encoded during training
    pad_length = 120
    encoded = tokenizer.selfies_tokenizer.selfies_to_indices(selfies)
    # Truncate and pad to fixed length
    pad_idx_selfies = tokenizer.selfies_tokenizer.symbol_to_index[tokenizer.selfies_tokenizer.PAD_TOKEN]
    encoded = encoded[:pad_length] + [pad_idx_selfies] * max(0, pad_length - len(encoded))
    encoded = encoded[:pad_length]  # Ensure exactly pad_length
    selfies_tensor = torch.tensor(encoded, dtype=torch.long, device=device_obj)
    seq_len = selfies_tensor.size(0)

    # Prepare output tensors
    selfies_batch = selfies_tensor.unsqueeze(0).expand(num_samples, seq_len).clone()
    properties = torch.full((num_samples, nprops), pad_idx, dtype=torch.long, device=device_obj)
    values = torch.full((num_samples, nprops), pad_idx, dtype=torch.long, device=device_obj)
    mask = torch.zeros((num_samples, nprops), dtype=torch.bool, device=device_obj)

    # Pre-convert context to lists for faster sampling
    ctx_props_list = context['property_token'].tolist()
    ctx_vals_list = context['value'].tolist()
    num_ctx = len(ctx_props_list)

    # Generate samples with different property combinations
    for i in range(num_samples):
        # Sample and shuffle context properties (more efficient without pandas sampling)
        sample_size = min(nprops - 1, num_ctx)
        indices = random.sample(range(num_ctx), sample_size)
        random.shuffle(indices)

        ctx_props = [ctx_props_list[idx] for idx in indices]
        ctx_vals = [ctx_vals_list[idx] for idx in indices]

        # Place target at end
        all_props = ctx_props + [property_token]
        all_vals = ctx_vals + [true_value]

        # Fill tensors (batch operation)
        properties[i, :nprops] = torch.tensor(all_props, dtype=torch.long, device=device_obj)
        values[i, :nprops] = torch.tensor(all_vals, dtype=torch.long, device=device_obj)
        mask[i, :nprops] = True

    return selfies_batch, properties, values, mask, true_value


def iterate_holdout_split(
    split_id: int,
    tokenizer,
    nprops: int,
    num_samples_per_pair: int = 50,
    db_path: str = "brick/cvae.sqlite",
    device: str = 'cpu',
    limit: Optional[int] = None,
) -> Generator[Dict, None, None]:
    """
    Iterate through all holdout pairs in a split and generate samples.

    Args:
        split_id: Bootstrap split ID (0-4)
        tokenizer: SelfiesPropertyValTokenizer instance
        nprops: Number of properties per sample
        num_samples_per_pair: Samples to generate per (selfies, property) pair
        db_path: Path to SQLite database
        device: Device to place tensors on
        limit: Optional limit on pairs to process

    Yields:
        Dict with keys: selfies, property_token, true_value, selfies_batch, properties, values, mask

    Example:
        >>> for batch in iterate_holdout_split(split_id=0, tokenizer=tok, nprops=5):
        ...     logits = model(batch['selfies_batch'], batch['properties'],
        ...                    batch['values'], batch['mask'])
        ...     # Process predictions...
    """
    conn = sqlite3.connect(db_path)
    query = "SELECT selfies, property_token, value FROM holdout_samples WHERE split_id = ?"
    holdout = pd.read_sql_query(query, conn, params=(split_id,))
    conn.close()

    print(f"Split {split_id}: {len(holdout)} holdout pairs")

    count = 0
    for _, row in holdout.iterrows():
        try:
            selfies_batch, props, vals, mask, true_val = generate_holdout_samples(
                selfies=row['selfies'],
                property_token=int(row['property_token']),
                tokenizer=tokenizer,
                nprops=nprops,
                num_samples=num_samples_per_pair,
                db_path=db_path,
                device=device,
            )

            yield {
                'selfies': row['selfies'],
                'property_token': int(row['property_token']),
                'true_value': true_val,
                'selfies_batch': selfies_batch,
                'properties': props,
                'values': vals,
                'mask': mask,
            }

            count += 1
            if limit is not None and count >= limit:
                return

        except Exception as e:
            print(f"Warning: Skipping ({row['selfies']}, {row['property_token']}): {e}")
            continue
