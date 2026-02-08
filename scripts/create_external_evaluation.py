#!/usr/bin/env python3
"""
Create external evaluation dataset from TDC data.
Find compounds NOT in ToxTransformer training data for unbiased evaluation.
"""
import pandas as pd
import sqlite3
from rdkit import Chem
from rdkit.Chem.inchi import MolFromInchi, MolToInchi
import sys

DB_PATH = "cache/build_sqlite/cvae.sqlite"
TDC_BASE = "/mnt/ssd/biobricks/biobricks-ai/chempile-tox/213e03f6bb303f2e3d920041f16e2ef161e74860/brick"

def smiles_to_inchi(smiles):
    """Convert SMILES to InChI."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        return Chem.MolToInchi(mol)
    except:
        return None

def get_tt_training_inchis():
    """Get all InChIs from ToxTransformer training data."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT inchi FROM activity WHERE inchi IS NOT NULL")
    inchis = set(row[0] for row in cursor.fetchall())
    conn.close()
    return inchis

def find_external_compounds(tdc_file, dataset_name, limit=500):
    """Find compounds in TDC data that are NOT in TT training data."""
    print(f"\n=== Processing {dataset_name} ===")

    # Load TDC data
    df = pd.read_parquet(tdc_file)
    print(f"  Total compounds: {len(df)}")

    # Convert SMILES to InChI
    print("  Converting SMILES to InChI...")
    df['inchi'] = df['smiles'].apply(smiles_to_inchi)
    df = df[df['inchi'].notna()]
    print(f"  Valid InChI conversions: {len(df)}")

    # Get TT training InChIs
    tt_inchis = get_tt_training_inchis()
    print(f"  TT training compounds: {len(tt_inchis)}")

    # Find external compounds (NOT in training)
    df['in_training'] = df['inchi'].isin(tt_inchis)
    external = df[~df['in_training']].copy()
    internal = df[df['in_training']].copy()

    print(f"  In TT training: {len(internal)}")
    print(f"  External (NOT in training): {len(external)}")

    # Sample if too many
    if len(external) > limit:
        external = external.sample(n=limit, random_state=42)
        print(f"  Sampled to: {len(external)}")

    return external, internal

def main():
    print("Creating External Evaluation Datasets")
    print("=" * 60)

    # Process Ames mutagenicity
    ames_external, ames_internal = find_external_compounds(
        f"{TDC_BASE}/tdc_ames.parquet", "Ames Mutagenicity", limit=500
    )

    # Process Carcinogenicity
    carc_external, carc_internal = find_external_compounds(
        f"{TDC_BASE}/tdc_carcinogens_lagunin.parquet", "Carcinogenicity", limit=300
    )

    # Process LD50
    ld50_external, ld50_internal = find_external_compounds(
        f"{TDC_BASE}/tdc_ld50_zhu.parquet", "LD50", limit=500
    )

    # Save results
    output_dir = "cache/external_evaluation"
    import os
    os.makedirs(output_dir, exist_ok=True)

    ames_external[['inchi', 'smiles', 'label']].to_parquet(f"{output_dir}/ames_external.parquet")
    carc_external[['inchi', 'smiles', 'label']].to_parquet(f"{output_dir}/carcinogens_external.parquet")
    ld50_external[['inchi', 'smiles', 'label']].to_parquet(f"{output_dir}/ld50_external.parquet")

    print("\n" + "=" * 60)
    print("External evaluation datasets saved to cache/external_evaluation/")
    print(f"  - ames_external.parquet: {len(ames_external)} compounds")
    print(f"  - carcinogens_external.parquet: {len(carc_external)} compounds")
    print(f"  - ld50_external.parquet: {len(ld50_external)} compounds")

    # Also save a combined summary
    summary = {
        'ames': {'external': len(ames_external), 'internal': len(ames_internal)},
        'carcinogens': {'external': len(carc_external), 'internal': len(carc_internal)},
        'ld50': {'external': len(ld50_external), 'internal': len(ld50_internal)},
    }

    # TT property tokens for each endpoint
    property_mapping = {
        'ames': [2508, 1799, 3830, 2863],  # mutagenicity properties
        'carcinogens': [3574, 3128, 3291, 2881, 3191],  # carcinogenicity properties
        'ld50': [1181, 1152],  # LD50 properties
    }

    import json
    with open(f"{output_dir}/evaluation_config.json", 'w') as f:
        json.dump({
            'summary': summary,
            'property_mapping': property_mapping
        }, f, indent=2)

    print(f"\nProperty mapping saved to evaluation_config.json")
    print("\nReady for ToxTransformer evaluation!")

if __name__ == "__main__":
    main()
