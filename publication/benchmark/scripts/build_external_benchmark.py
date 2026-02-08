#!/usr/bin/env python3
"""
Build comprehensive external evaluation benchmark for ToxTransformer.

This script:
1. Scans all BioBricks datasets for chemical-property-value data
2. Converts SMILES to InChI
3. Identifies compounds NOT in ToxTransformer training data
4. Creates a large, diverse external evaluation dataset

Target: 10k+ compounds across 100+ properties for publication-quality benchmark.
"""

import os
import sys
import json
import glob
import hashlib
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# RDKit for SMILES -> InChI conversion
from rdkit import Chem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

# Configuration
BIOBRICKS_DIR = "/mnt/ssd/biobricks/biobricks-ai"
TT_DB_PATH = "/mnt/ssd/git/chemprop-transformer/cache/build_sqlite/cvae.sqlite"
OUTPUT_DIR = "/mnt/ssd/git/chemprop-transformer/publication/benchmark/data"

# Datasets to scan - prioritize toxicity/ADMET datasets
PRIORITY_DATASETS = [
    # TDC ADMET datasets (high priority - curated benchmarks)
    "tdc-admet",
    "tdc-dili",
    "tdc-herg-central",
    "tdc-clearance-hepatocyte",
    "tdc-clearance-microsome",
    "tdc-half-life",
    "tdc-ppbr",

    # ChemPile toxicity (aggregated)
    "chempile-tox",

    # Specific toxicity datasets
    "unitox",
    "dilirank",
    "skin-reaction",
    "skin-sens-multiassay",
    "pampa-ncats",
    "qsar-aquatic-tox",
    "pfas-tox",
    # "toxric",  # Skip - 82M records, too large
    # "toxric-figshare",  # Skip - duplicate of toxric
    "openfoodtox",
    "nura",  # nanotoxicity
    "t3db",  # toxin database
    "rtecs",  # toxic effects
    "ames-benchmark",
    "ames-qsar",
    "carcinogens-lagunin",

    # ADMET/PK datasets
    "admet-huggingface",
    "computational-adme",
    "bioavailability-ma",
    "solubility-aqsoldb",
    "esol",
    # "b3db",  # Skip - 11.8M records, too large
    "bbbp",

    # Drug safety
    "adrecs",
    "faers",
    "sider",
    "twosides",
    "onsides",
    "smiles-adr",
    "withdrawn",

    # Ecotoxicity
    "ecotox",
    "adore-ecotox",
    "aquatic-moa-ecotox",
    "comptox-zebrafish",
    "concawe-petrotox",
    "qsar-aquatic-tox",

    # Bioactivity (for comparison)
    # "excape-db",  # Very large
    # "papyrus",  # Very large
    "pharmabench",
    # "pcba",  # Very large

    # Additional curated datasets
    "cpdb",  # Carcinogenicity
    "eutoxrisk-temposeq",  # ToxCast-related
    "drugmatrix",  # FDA drug toxicity
    "tggates",  # Toxicogenomics
    "deduct",  # Drug-drug interactions
    "cyp450",  # CYP metabolism
    "cosmos-ng",  # Cosmetics safety
    "ntp-roc",  # NTP carcinogenicity
    "edkb",  # Endocrine disruptors
]

def smiles_to_inchi(smiles: str) -> Optional[str]:
    """Convert SMILES to InChI."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchi(mol)
    except:
        return None

def get_tt_training_inchis() -> set:
    """Get all InChIs from ToxTransformer training data."""
    print("Loading ToxTransformer training InChIs...")
    conn = sqlite3.connect(TT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT inchi FROM activity WHERE inchi IS NOT NULL")
    inchis = set(row[0] for row in cursor.fetchall())
    conn.close()
    print(f"  Loaded {len(inchis):,} training InChIs")
    return inchis

def get_tt_properties() -> Dict[int, str]:
    """Get all ToxTransformer property tokens and titles."""
    conn = sqlite3.connect(TT_DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT property_token, title FROM property WHERE property_token IS NOT NULL")
    props = {int(row[0]): row[1] for row in cursor.fetchall() if row[0] is not None}
    conn.close()
    return props

def find_brick_dir(dataset_name: str) -> Optional[Path]:
    """Find the brick directory for a dataset."""
    dataset_path = Path(BIOBRICKS_DIR) / dataset_name
    if not dataset_path.exists():
        return None

    # Find the hash subdirectory
    for subdir in dataset_path.iterdir():
        if subdir.is_dir() and not subdir.name.startswith('.'):
            brick_path = subdir / "brick"
            if brick_path.exists():
                return brick_path
    return None

def scan_parquet_files(brick_dir: Path) -> List[Dict]:
    """Scan parquet files in a brick directory for chemical data."""
    results = []

    for pq_file in brick_dir.glob("*.parquet"):
        try:
            df = pd.read_parquet(pq_file)

            # Look for SMILES/InChI columns
            smiles_cols = [c for c in df.columns if 'smiles' in c.lower() or c.lower() == 'smiles']
            inchi_cols = [c for c in df.columns if 'inchi' in c.lower()]

            # Look for label/value columns
            label_cols = [c for c in df.columns if any(x in c.lower() for x in
                ['label', 'value', 'activity', 'outcome', 'target', 'class', 'result', 'response'])]

            if (smiles_cols or inchi_cols) and label_cols:
                results.append({
                    'file': pq_file,
                    'smiles_cols': smiles_cols,
                    'inchi_cols': inchi_cols,
                    'label_cols': label_cols,
                    'shape': df.shape,
                    'columns': list(df.columns)
                })
        except Exception as e:
            pass  # Skip files that can't be read

    return results

def extract_dataset(brick_dir: Path, dataset_name: str, max_records_per_property: int = 50000, max_total_records: int = 500000) -> pd.DataFrame:
    """Extract chemical-property-value data from a dataset."""
    all_data = []

    for pq_file in brick_dir.glob("*.parquet"):
        try:
            # For very large files, read in chunks or sample
            file_size = pq_file.stat().st_size
            if file_size > 2e9:  # > 2GB - skip entirely
                print(f"      Skipping very large file {pq_file.name} ({file_size/1e9:.1f}GB)")
                continue
            elif file_size > 500e6:  # > 500MB - sample
                print(f"      Large file {pq_file.name} ({file_size/1e6:.0f}MB), sampling...")
                df = pd.read_parquet(pq_file)
                if len(df) > max_total_records:
                    df = df.sample(n=max_total_records, random_state=42)
            else:
                df = pd.read_parquet(pq_file)
                if len(df) > max_total_records:
                    print(f"      Large dataset ({len(df):,} rows), sampling to {max_total_records:,}...")
                    df = df.sample(n=max_total_records, random_state=42)

            # Find chemical identifier column
            smiles_col = None
            inchi_col = None
            for c in df.columns:
                if 'smiles' in c.lower() or c.lower() == 'smiles':
                    smiles_col = c
                    break
            for c in df.columns:
                if 'inchi' in c.lower() and 'key' not in c.lower():
                    inchi_col = c
                    break

            if not smiles_col and not inchi_col:
                continue

            # Find label columns (numeric columns that aren't identifiers)
            exclude_cols = {'smiles', 'inchi', 'inchikey', 'name', 'id', 'cas', 'cid', 'chembl_id'}
            label_cols = []
            for c in df.columns:
                if c.lower() in exclude_cols:
                    continue
                if df[c].dtype in ['float64', 'int64', 'float32', 'int32']:
                    # Check if it looks like a label (0/1 or continuous values)
                    unique_vals = df[c].dropna().unique()
                    if len(unique_vals) > 1 and len(unique_vals) < len(df) * 0.9:
                        label_cols.append(c)

            if not label_cols:
                continue

            # Extract data
            file_stem = pq_file.stem
            for label_col in label_cols:
                property_name = f"{dataset_name}/{file_stem}/{label_col}"

                temp_df = df[[smiles_col or inchi_col, label_col]].copy()
                temp_df.columns = ['compound', 'value']
                temp_df = temp_df.dropna()

                if len(temp_df) < 10:  # Skip tiny datasets
                    continue

                temp_df['property'] = property_name
                temp_df['source'] = dataset_name
                temp_df['is_smiles'] = smiles_col is not None

                # Determine if binary or continuous
                unique_vals = temp_df['value'].unique()
                is_binary = len(unique_vals) == 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0})
                temp_df['is_binary'] = is_binary

                all_data.append(temp_df)

        except Exception as e:
            continue

    if all_data:
        return pd.concat(all_data, ignore_index=True)
    return pd.DataFrame()

def process_all_datasets(tt_inchis: set) -> pd.DataFrame:
    """Process all priority datasets and extract external compounds."""
    all_external = []
    dataset_stats = []

    print("\nProcessing BioBricks datasets...")

    for dataset_name in PRIORITY_DATASETS:
        brick_dir = find_brick_dir(dataset_name)
        if not brick_dir:
            continue

        print(f"\n  Processing {dataset_name}...")
        df = extract_dataset(brick_dir, dataset_name)

        if df.empty:
            print(f"    No suitable data found")
            continue

        print(f"    Found {len(df):,} records across {df['property'].nunique()} properties")

        # Convert to InChI
        if df['is_smiles'].iloc[0]:
            print(f"    Converting SMILES to InChI...")
            df['inchi'] = df['compound'].apply(smiles_to_inchi)
            df = df[df['inchi'].notna()]
            print(f"    Valid conversions: {len(df):,}")
        else:
            df['inchi'] = df['compound']

        if df.empty:
            continue

        # Check overlap with training data
        df['in_training'] = df['inchi'].isin(tt_inchis)
        external = df[~df['in_training']].copy()
        internal = df[df['in_training']]

        print(f"    In TT training: {len(internal):,} ({100*len(internal)/len(df):.1f}%)")
        print(f"    External: {len(external):,} ({100*len(external)/len(df):.1f}%)")

        if len(external) > 0:
            all_external.append(external)

            # Track stats
            for prop in external['property'].unique():
                prop_df = external[external['property'] == prop]
                dataset_stats.append({
                    'dataset': dataset_name,
                    'property': prop,
                    'n_compounds': len(prop_df),
                    'n_unique_compounds': prop_df['inchi'].nunique(),
                    'is_binary': prop_df['is_binary'].iloc[0],
                    'value_range': f"{prop_df['value'].min():.3f}-{prop_df['value'].max():.3f}"
                })

    if all_external:
        combined = pd.concat(all_external, ignore_index=True)
        stats_df = pd.DataFrame(dataset_stats)
        return combined, stats_df

    return pd.DataFrame(), pd.DataFrame()

def create_benchmark_splits(df: pd.DataFrame, stats_df: pd.DataFrame) -> Dict:
    """Create benchmark data splits and metadata."""

    # Filter to binary classification properties (primary focus)
    binary_df = df[df['is_binary']].copy()
    continuous_df = df[~df['is_binary']].copy()

    print(f"\nBenchmark Summary:")
    print(f"  Total external records: {len(df):,}")
    print(f"  Unique compounds: {df['inchi'].nunique():,}")
    print(f"  Unique properties: {df['property'].nunique()}")
    print(f"  Binary classification: {len(binary_df):,} records, {binary_df['property'].nunique()} properties")
    print(f"  Continuous: {len(continuous_df):,} records, {continuous_df['property'].nunique()} properties")

    # Create property mapping for ToxTransformer
    tt_props = get_tt_properties()

    # Try to match external properties to TT properties
    property_mapping = {}
    for prop in df['property'].unique():
        prop_lower = prop.lower()
        matches = []
        for tt_token, tt_title in tt_props.items():
            if tt_title:
                tt_lower = tt_title.lower()
                # Simple keyword matching
                keywords = ['mutagen', 'carcinogen', 'toxic', 'lethal', 'ames', 'herg',
                           'dili', 'bbb', 'permeab', 'solub', 'cyp', 'pgp', 'bioavail']
                for kw in keywords:
                    if kw in prop_lower and kw in tt_lower:
                        matches.append((tt_token, tt_title))
                        break
        if matches:
            property_mapping[prop] = matches[:5]  # Top 5 matches

    return {
        'binary_df': binary_df,
        'continuous_df': continuous_df,
        'stats_df': stats_df,
        'property_mapping': property_mapping,
        'tt_properties': tt_props
    }

def save_benchmark(data: Dict, output_dir: str):
    """Save benchmark data to files."""
    os.makedirs(output_dir, exist_ok=True)

    # Save main data files
    data['binary_df'][['inchi', 'value', 'property', 'source']].to_parquet(
        f"{output_dir}/external_binary_benchmark.parquet", index=False)
    data['continuous_df'][['inchi', 'value', 'property', 'source']].to_parquet(
        f"{output_dir}/external_continuous_benchmark.parquet", index=False)
    data['stats_df'].to_csv(f"{output_dir}/property_statistics.csv", index=False)

    # Save metadata
    metadata = {
        'created': datetime.now().isoformat(),
        'total_compounds': int(data['binary_df']['inchi'].nunique() + data['continuous_df']['inchi'].nunique()),
        'total_properties': int(data['binary_df']['property'].nunique() + data['continuous_df']['property'].nunique()),
        'binary_classification': {
            'n_records': int(len(data['binary_df'])),
            'n_compounds': int(data['binary_df']['inchi'].nunique()),
            'n_properties': int(data['binary_df']['property'].nunique()),
            'sources': list(data['binary_df']['source'].unique())
        },
        'continuous_regression': {
            'n_records': int(len(data['continuous_df'])),
            'n_compounds': int(data['continuous_df']['inchi'].nunique()),
            'n_properties': int(data['continuous_df']['property'].nunique()),
            'sources': list(data['continuous_df']['source'].unique())
        },
        'property_mapping': {k: [(int(t), n) for t, n in v] for k, v in data['property_mapping'].items()}
    }

    with open(f"{output_dir}/benchmark_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    # Save a sample for quick inspection
    sample_binary = data['binary_df'].groupby('property').apply(
        lambda x: x.sample(n=min(10, len(x)), random_state=42)
    ).reset_index(drop=True)
    sample_binary.to_csv(f"{output_dir}/sample_binary.csv", index=False)

    print(f"\nSaved benchmark data to {output_dir}/")
    print(f"  - external_binary_benchmark.parquet ({len(data['binary_df']):,} records)")
    print(f"  - external_continuous_benchmark.parquet ({len(data['continuous_df']):,} records)")
    print(f"  - property_statistics.csv")
    print(f"  - benchmark_metadata.json")

def main():
    print("=" * 70)
    print("Building ToxTransformer External Evaluation Benchmark")
    print("=" * 70)

    # Load TT training data
    tt_inchis = get_tt_training_inchis()

    # Process all datasets
    external_df, stats_df = process_all_datasets(tt_inchis)

    if external_df.empty:
        print("\nNo external data found!")
        return

    # Create benchmark splits
    data = create_benchmark_splits(external_df, stats_df)

    # Save benchmark
    save_benchmark(data, OUTPUT_DIR)

    print("\n" + "=" * 70)
    print("Benchmark creation complete!")
    print("=" * 70)

if __name__ == "__main__":
    main()
