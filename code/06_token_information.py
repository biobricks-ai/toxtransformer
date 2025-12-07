#!/usr/bin/env python
"""
Calculate mutual information between property tokens using map-reduce over tensor files.

This script computes pairwise mutual information (MI) between property tokens using
a fast map-reduce approach that processes the pre-built tensor dataset in parallel.

For each molecule, we count co-occurrences of property value pairs:
- n_00: both properties are 0
- n_01: first is 0, second is 1
- n_10: first is 1, second is 0
- n_11: both are 1

Then compute MI from aggregated counts.

Outputs:
- cache/token_information/pairwise_mi.parquet: Pairwise MI for property pairs
- cache/token_information/source_mi_summary.parquet: Within-source MI statistics
- cache/token_information/property_statistics.parquet: Per-property statistics

Usage:
    python code/2_2_1_token_information.py

    # To compute only within-source MI (faster):
    python code/2_2_1_token_information.py --within-source-only

    # Specify number of workers:
    python code/2_2_1_token_information.py --workers 64
"""

import argparse
import logging
import pathlib
import sqlite3
import numpy as np
import pandas as pd
import torch
from collections import defaultdict
from typing import Dict, Tuple, List
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import glob

# Configuration
DB_PATH = "brick/cvae.sqlite"
TENSOR_DIR = "cache/build_tensordataset/final_tensors"
OUTDIR = pathlib.Path("cache/token_information")
MIN_COOCCURRENCE = 50  # Minimum molecules with both properties to compute MI


def setup_logging(outdir: pathlib.Path):
    """Configure logging."""
    logdir = outdir / "logs"
    logdir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(logdir / 'token_information.log', mode='w'),
            logging.StreamHandler()
        ]
    )


def load_source_mapping(db_path: str) -> Dict[int, str]:
    """Load source_id to source name mapping."""
    conn = sqlite3.connect(db_path)
    sources = pd.read_sql("SELECT source_id, source FROM source", conn)
    conn.close()
    return dict(zip(sources['source_id'], sources['source']))


def load_property_source_mapping(db_path: str) -> Dict[int, int]:
    """Load property_token to source_id mapping."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT DISTINCT property_token, source_id
        FROM activity
    """
    mapping = pd.read_sql(query, conn)
    conn.close()
    return dict(zip(mapping['property_token'], mapping['source_id']))


def process_tensor_file(
    filepath: str,
    prop_to_source: Dict[int, int],
    within_source_only: bool = False,
) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[int, List[int]]]:
    """
    Process a single tensor file and return pair counts and property stats.

    Returns:
        pair_counts: Dict mapping (prop1, prop2) -> [n_00, n_01, n_10, n_11]
        prop_stats: Dict mapping prop -> [n_total, n_positive]
    """
    data = torch.load(filepath, weights_only=True)
    properties = data['properties']  # [batch, max_props]
    values = data['values']  # [batch, max_props]

    batch_size = properties.shape[0]

    # Accumulate counts
    pair_counts = defaultdict(lambda: [0, 0, 0, 0])
    prop_stats = defaultdict(lambda: [0, 0])  # [n_total, n_positive]

    for i in range(batch_size):
        # Get valid properties for this molecule (not -1)
        props = properties[i]
        vals = values[i]
        mask = props >= 0

        valid_props = props[mask].numpy()
        valid_vals = vals[mask].numpy()

        n_props = len(valid_props)
        if n_props < 2:
            continue

        # Update property stats
        for p, v in zip(valid_props, valid_vals):
            p = int(p)
            v = int(v)
            prop_stats[p][0] += 1
            prop_stats[p][1] += v

        # Enumerate all pairs
        for j in range(n_props):
            p1 = int(valid_props[j])
            v1 = int(valid_vals[j])
            s1 = prop_to_source.get(p1, -1)

            for k in range(j + 1, n_props):
                p2 = int(valid_props[k])
                v2 = int(valid_vals[k])
                s2 = prop_to_source.get(p2, -1)

                # Skip cross-source pairs if within_source_only
                if within_source_only and s1 != s2:
                    continue

                # Canonical ordering (smaller prop first)
                if p1 > p2:
                    p1, p2 = p2, p1
                    v1, v2 = v2, v1

                # Update count based on value combination
                idx = v1 * 2 + v2  # 0=00, 1=01, 2=10, 3=11
                pair_counts[(p1, p2)][idx] += 1

    return dict(pair_counts), dict(prop_stats)


def merge_counts(
    all_pair_counts: List[Dict],
    all_prop_stats: List[Dict],
) -> Tuple[Dict[Tuple[int, int], List[int]], Dict[int, List[int]]]:
    """Merge counts from multiple workers."""
    merged_pairs = defaultdict(lambda: [0, 0, 0, 0])
    merged_props = defaultdict(lambda: [0, 0])

    for pair_counts in all_pair_counts:
        for key, counts in pair_counts.items():
            for i in range(4):
                merged_pairs[key][i] += counts[i]

    for prop_stats in all_prop_stats:
        for key, stats in prop_stats.items():
            merged_props[key][0] += stats[0]
            merged_props[key][1] += stats[1]

    return dict(merged_pairs), dict(merged_props)


def compute_mi_from_counts(counts: List[int]) -> Tuple[float, float, float]:
    """
    Compute MI, NMI, and entropy from contingency table counts.

    Args:
        counts: [n_00, n_01, n_10, n_11]

    Returns:
        (mi, nmi, n_total)
    """
    n_00, n_01, n_10, n_11 = counts
    n_total = n_00 + n_01 + n_10 + n_11

    if n_total == 0:
        return 0.0, 0.0, 0

    # Joint probabilities
    p_00 = n_00 / n_total
    p_01 = n_01 / n_total
    p_10 = n_10 / n_total
    p_11 = n_11 / n_total

    # Marginals
    p_x0 = p_00 + p_01
    p_x1 = p_10 + p_11
    p_y0 = p_00 + p_10
    p_y1 = p_01 + p_11

    # Mutual information
    eps = 1e-10
    mi = 0.0
    for p_xy, p_x, p_y in [
        (p_00, p_x0, p_y0),
        (p_01, p_x0, p_y1),
        (p_10, p_x1, p_y0),
        (p_11, p_x1, p_y1),
    ]:
        if p_xy > eps and p_x > eps and p_y > eps:
            mi += p_xy * np.log2(p_xy / (p_x * p_y))

    # Entropies for NMI
    h_x = 0.0
    if p_x0 > eps:
        h_x -= p_x0 * np.log2(p_x0)
    if p_x1 > eps:
        h_x -= p_x1 * np.log2(p_x1)

    h_y = 0.0
    if p_y0 > eps:
        h_y -= p_y0 * np.log2(p_y0)
    if p_y1 > eps:
        h_y -= p_y1 * np.log2(p_y1)

    # Normalized MI
    nmi = 2 * mi / (h_x + h_y) if (h_x + h_y) > eps else 0.0

    return mi, nmi, n_total


def compute_property_entropy(n_total: int, n_positive: int) -> float:
    """Compute entropy of a binary property."""
    if n_total == 0:
        return 0.0

    p1 = n_positive / n_total
    p0 = 1 - p1

    h = 0.0
    eps = 1e-10
    if p0 > eps:
        h -= p0 * np.log2(p0)
    if p1 > eps:
        h -= p1 * np.log2(p1)

    return h


def main():
    parser = argparse.ArgumentParser(
        description='Calculate mutual information between property tokens (map-reduce)'
    )
    parser.add_argument('--within-source-only', action='store_true',
                       help='Only compute within-source MI (faster)')
    parser.add_argument('--min-cooccurrence', type=int, default=MIN_COOCCURRENCE,
                       help=f'Minimum co-occurrence count (default: {MIN_COOCCURRENCE})')
    parser.add_argument('--db-path', type=str, default=DB_PATH,
                       help=f'Path to SQLite database (default: {DB_PATH})')
    parser.add_argument('--tensor-dir', type=str, default=TENSOR_DIR,
                       help=f'Path to tensor directory (default: {TENSOR_DIR})')
    parser.add_argument('--workers', type=int, default=64,
                       help='Number of parallel workers (default: 64)')
    args = parser.parse_args()

    # Setup
    OUTDIR.mkdir(parents=True, exist_ok=True)
    setup_logging(OUTDIR)

    logging.info("="*60)
    logging.info("Token Information Analysis (Map-Reduce)")
    logging.info("="*60)
    logging.info(f"Database: {args.db_path}")
    logging.info(f"Tensor directory: {args.tensor_dir}")
    logging.info(f"Within-source only: {args.within_source_only}")
    logging.info(f"Min co-occurrence: {args.min_cooccurrence}")
    logging.info(f"Workers: {args.workers}")

    # Load mappings
    logging.info("Loading property-source mapping...")
    source_mapping = load_source_mapping(args.db_path)
    prop_to_source = load_property_source_mapping(args.db_path)
    logging.info(f"  Loaded {len(prop_to_source)} property-source mappings")
    logging.info(f"  Sources: {list(source_mapping.values())}")

    # Find tensor files
    tensor_files = sorted(glob.glob(f"{args.tensor_dir}/*.pt"))
    logging.info(f"Found {len(tensor_files)} tensor files")

    if len(tensor_files) == 0:
        logging.error(f"No tensor files found in {args.tensor_dir}")
        return

    # Process tensor files in parallel
    logging.info("Processing tensor files...")

    all_pair_counts = []
    all_prop_stats = []

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                process_tensor_file, f, prop_to_source, args.within_source_only
            ): f for f in tensor_files
        }

        for future in tqdm(as_completed(futures), total=len(tensor_files), desc="Files"):
            pair_counts, prop_stats = future.result()
            all_pair_counts.append(pair_counts)
            all_prop_stats.append(prop_stats)

    # Merge counts
    logging.info("Merging counts...")
    merged_pairs, merged_props = merge_counts(all_pair_counts, all_prop_stats)
    logging.info(f"  Total property pairs: {len(merged_pairs):,}")
    logging.info(f"  Total properties: {len(merged_props):,}")

    # Compute property statistics
    logging.info("Computing property statistics...")
    prop_stats_list = []
    for prop, (n_total, n_positive) in merged_props.items():
        source_id = prop_to_source.get(prop, -1)
        n_negative = n_total - n_positive
        entropy = compute_property_entropy(n_total, n_positive)
        positive_rate = n_positive / n_total if n_total > 0 else 0

        prop_stats_list.append({
            'property_token': prop,
            'source_id': source_id,
            'n_molecules': n_total,
            'n_positive': n_positive,
            'n_negative': n_negative,
            'positive_rate': positive_rate,
            'entropy': entropy,
        })

    prop_stats_df = pd.DataFrame(prop_stats_list)
    prop_stats_df['source'] = prop_stats_df['source_id'].map(source_mapping)

    prop_stats_path = OUTDIR / 'property_statistics.parquet'
    prop_stats_df.to_parquet(prop_stats_path, index=False)
    logging.info(f"Saved property statistics to {prop_stats_path}")

    # Show property count by source
    source_prop_counts = prop_stats_df.groupby('source').agg({
        'property_token': 'count',
        'n_molecules': ['mean', 'sum'],
        'entropy': 'mean',
    }).reset_index()
    source_prop_counts.columns = ['source', 'n_properties', 'mean_molecules', 'total_molecule_props', 'mean_entropy']
    source_prop_counts = source_prop_counts.sort_values('n_properties', ascending=False)
    logging.info(f"\nProperty counts by source:\n{source_prop_counts.to_string(index=False)}")

    # Compute pairwise MI
    logging.info("Computing pairwise MI...")
    mi_results = []

    for (p1, p2), counts in tqdm(merged_pairs.items(), desc="Computing MI"):
        n_total = sum(counts)

        if n_total < args.min_cooccurrence:
            continue

        mi, nmi, _ = compute_mi_from_counts(counts)

        s1 = prop_to_source.get(p1, -1)
        s2 = prop_to_source.get(p2, -1)

        mi_results.append({
            'prop1': p1,
            'prop2': p2,
            'mi': mi,
            'nmi': nmi,
            'n_cooccur': n_total,
            'source1': s1,
            'source2': s2,
        })

    logging.info(f"Computed MI for {len(mi_results):,} property pairs")

    if len(mi_results) > 0:
        pairwise_df = pd.DataFrame(mi_results)

        pairwise_path = OUTDIR / 'pairwise_mi.parquet'
        pairwise_df.to_parquet(pairwise_path, index=False)
        logging.info(f"Saved pairwise MI to {pairwise_path}")

        # Compute source summary (within-source only)
        logging.info("Computing within-source MI summary...")
        within_source = pairwise_df[pairwise_df['source1'] == pairwise_df['source2']].copy()

        if len(within_source) > 0:
            summary = within_source.groupby('source1').agg({
                'mi': ['mean', 'std', 'max', 'count'],
                'nmi': ['mean', 'std', 'max'],
                'n_cooccur': ['mean', 'min', 'max'],
            }).reset_index()

            summary.columns = [
                'source_id',
                'mean_mi', 'std_mi', 'max_mi', 'n_pairs',
                'mean_nmi', 'std_nmi', 'max_nmi',
                'mean_cooccur', 'min_cooccur', 'max_cooccur',
            ]

            summary['source'] = summary['source_id'].map(source_mapping)
            summary = summary.sort_values('mean_mi', ascending=False)

            summary_path = OUTDIR / 'source_mi_summary.parquet'
            summary.to_parquet(summary_path, index=False)
            logging.info(f"Saved source MI summary to {summary_path}")

            logging.info(f"\n{'='*60}")
            logging.info("WITHIN-SOURCE MUTUAL INFORMATION SUMMARY")
            logging.info(f"{'='*60}")
            logging.info(f"\n{summary.to_string(index=False)}")

            # Highlight high-MI sources
            high_mi_sources = summary[summary['mean_mi'] > 0.1]
            if len(high_mi_sources) > 0:
                logging.info(f"\n{'!'*60}")
                logging.info("WARNING: Sources with high within-source MI (potential task leakage):")
                logging.info(f"{'!'*60}")
                for _, row in high_mi_sources.iterrows():
                    logging.info(f"  {row['source']}: mean_MI={row['mean_mi']:.3f}, "
                               f"mean_NMI={row['mean_nmi']:.3f}, n_pairs={row['n_pairs']}")

    logging.info(f"\n{'='*60}")
    logging.info("Token information analysis complete!")
    logging.info(f"Output directory: {OUTDIR}")
    logging.info(f"{'='*60}")


if __name__ == "__main__":
    main()
