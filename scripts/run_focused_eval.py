#!/usr/bin/env python3
"""
Focused evaluation on best external datasets with careful token matching.
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import sys
import os
from urllib.parse import quote
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

API_URL = os.environ.get("TOXTRANSFORMER_API", "http://136.111.102.10:6515")
DATA_DIR = "publication/benchmark/data"
MAX_WORKERS = 10
TIMEOUT = 120

# FOCUSED MAPPINGS: External dataset -> (tokens, description)
# Only include mappings where we're confident the tokens match the endpoint
FOCUSED_MAPPINGS = {
    # AMES mutagenicity - bacterial mutation
    ("admet-huggingface", "admet/AMES"): {
        "name": "Ames Mutagenicity (ADMET-HF)",
        "tokens": [1799, 2508, 2863, 3830],
        "desc": "Salmonella/bacterial mutagenicity assays"
    },
    ("ames-benchmark", "ames-benchmark/ames_cv_splits/ames_class"): {
        "name": "Ames Mutagenicity (Benchmark)",
        "tokens": [1799, 2508, 2863, 3830],
        "desc": "Salmonella/bacterial mutagenicity assays"
    },
    ("chempile-tox", "chempile-tox/tdc_ames/label"): {
        "name": "Ames Mutagenicity (TDC)",
        "tokens": [1799, 2508, 2863, 3830],
        "desc": "Salmonella/bacterial mutagenicity assays"
    },

    # hERG cardiac
    ("admet-huggingface", "admet/hERG"): {
        "name": "hERG Inhibition (ADMET-HF)",
        "tokens": [213, 229, 1252, 1438, 1866, 2105, 2547],
        "desc": "hERG potassium channel inhibition"
    },

    # CYP450 - isoform specific
    ("cyp450", "cyp450/cyp450_cyp1a2_combined/label"): {
        "name": "CYP1A2 Inhibition",
        "tokens": [1270, 1610, 5808, 5815],
        "desc": "CYP1A2-specific inhibition"
    },
    ("cyp450", "cyp450/cyp450_cyp2c9_combined/label"): {
        "name": "CYP2C9 Inhibition",
        "tokens": [701, 1582, 1854, 2219, 2800],
        "desc": "CYP2C9-specific inhibition"
    },
    ("cyp450", "cyp450/cyp450_cyp2c19_combined/label"): {
        "name": "CYP2C19 Inhibition",
        "tokens": [702, 1576, 2290],
        "desc": "CYP2C19-specific inhibition"
    },
    ("cyp450", "cyp450/cyp450_cyp2d6_combined/label"): {
        "name": "CYP2D6 Inhibition",
        "tokens": [1585, 2295],
        "desc": "CYP2D6-specific inhibition"
    },
    ("cyp450", "cyp450/cyp450_cyp3a4_combined/label"): {
        "name": "CYP3A4 Inhibition",
        "tokens": [1148, 1521, 1864, 2138],
        "desc": "CYP3A4-specific inhibition"
    },

    # Carcinogenicity
    ("carcinogens-lagunin", "carcinogens-lagunin/carcinogens_lagunin/label"): {
        "name": "Carcinogenicity (Lagunin)",
        "tokens": [2881, 3128, 3191, 3291, 3574],
        "desc": "Carcinogenic potency from CPDB"
    },
    ("admet-huggingface", "admet/Carcinogens_Lagunin"): {
        "name": "Carcinogenicity (ADMET-HF)",
        "tokens": [2881, 3128, 3191, 3291, 3574],
        "desc": "Carcinogenic potency"
    },

    # Skin sensitization
    ("skin-reaction", "skin-reaction/skin_reaction/label"): {
        "name": "Skin Reaction",
        "tokens": [],  # Need to find tokens
        "desc": "Skin sensitization"
    },

    # P-glycoprotein
    ("admet-huggingface", "admet/Pgp_Broccatelli"): {
        "name": "P-gp Substrate",
        "tokens": [],  # Need to find tokens
        "desc": "P-glycoprotein substrate"
    },

    # Zebrafish - these worked well
    ("comptox-zebrafish", "comptox-zebrafish/zebrafish_toxicity/mortality"): {
        "name": "Zebrafish Mortality",
        "tokens": [3418, 3455, 3487, 3488, 3489, 3490, 3492, 4075],
        "desc": "Zebrafish developmental toxicity"
    },
    ("comptox-zebrafish", "comptox-zebrafish/zebrafish_toxicity/activity_score"): {
        "name": "Zebrafish Activity",
        "tokens": [3418, 3455, 3487, 3488, 3489, 3490, 3492, 4075],
        "desc": "Zebrafish developmental toxicity"
    },

    # UniTox organ toxicity - try matching to general cytotoxicity
    ("unitox", "unitox/unitox_small_molecules/liver_toxicity_binary_rating_0_1"): {
        "name": "Liver Toxicity (UniTox)",
        "tokens": [4438, 1294, 1298, 1300, 1386, 1404],  # HepG2 viability tokens
        "desc": "HepG2 hepatocyte cytotoxicity"
    },
    ("unitox", "unitox/unitox_small_molecules/cardio_toxicity_binary_rating_0_1"): {
        "name": "Cardiotoxicity (UniTox)",
        "tokens": [213, 229, 1252, 2105],  # hERG as proxy
        "desc": "hERG/cardiac ion channel"
    },
}


def predict_with_retry(inchi, max_retries=2):
    for attempt in range(max_retries):
        try:
            url = f"{API_URL}/predict_all?inchi={quote(inchi, safe='')}"
            response = requests.get(url, timeout=TIMEOUT)
            if response.status_code == 200:
                return response.json()
        except:
            if attempt < max_retries - 1:
                time.sleep(1)
    return None


def batch_predict(inchis):
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(predict_with_retry, inchi): inchi for inchi in inchis}
        for i, future in enumerate(as_completed(futures)):
            inchi = futures[future]
            try:
                preds = future.result()
                if preds:
                    results[inchi] = preds
            except:
                pass
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(inchis)} processed")
    return results


def extract_pred(predictions, tokens):
    if not predictions:
        return None
    values = [p['value'] for p in predictions if p.get('property_token') in tokens
              and p.get('value') is not None and not np.isnan(p.get('value', float('nan')))]
    return np.mean(values) if values else None


def run_focused_eval():
    print("=" * 90)
    print("FOCUSED EXTERNAL EVALUATION")
    print("=" * 90)
    print(f"Started: {datetime.now()}")

    # Check API
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        print(f"API: OK")
    except:
        print("API unreachable!")
        return

    # Load data
    binary_df = pd.read_parquet(f"{DATA_DIR}/external_binary_benchmark.parquet")
    print(f"Loaded {len(binary_df):,} binary records")

    results = []

    for (source, prop_path), mapping in FOCUSED_MAPPINGS.items():
        tokens = mapping["tokens"]
        if not tokens:
            continue

        name = mapping["name"]

        # Find matching data
        mask = (binary_df['source'] == source) & (binary_df['property'] == prop_path)
        prop_df = binary_df[mask]

        if len(prop_df) < 20:
            # Try partial match
            mask = (binary_df['source'] == source) & (binary_df['property'].str.contains(prop_path.split('/')[-1], case=False))
            prop_df = binary_df[mask]

        if len(prop_df) < 20:
            print(f"\n{name}: Skipped (only {len(prop_df)} samples)")
            continue

        n_pos = int(prop_df['value'].sum())
        n_neg = len(prop_df) - n_pos

        # Balance classes
        pos_df = prop_df[prop_df['value'] == 1]
        neg_df = prop_df[prop_df['value'] == 0]
        n_sample = min(200, len(pos_df), len(neg_df))

        if n_sample < 10:
            print(f"\n{name}: Skipped (class imbalance: {n_pos} pos, {n_neg} neg)")
            continue

        sampled = pd.concat([
            pos_df.sample(n=n_sample, random_state=42),
            neg_df.sample(n=n_sample, random_state=42)
        ])

        print(f"\n{name}: Evaluating {len(sampled)} samples ({n_sample} pos, {n_sample} neg)...")

        # Get predictions
        inchis = sampled['inchi'].unique()
        preds_cache = batch_predict(inchis)

        y_true, y_pred = [], []
        for _, row in sampled.iterrows():
            if row['inchi'] not in preds_cache:
                continue
            pred_val = extract_pred(preds_cache[row['inchi']], tokens)
            if pred_val is None:
                continue
            y_true.append(int(row['value']))
            y_pred.append(pred_val)

        if len(y_true) < 20 or len(set(y_true)) < 2:
            print(f"  Insufficient valid predictions: {len(y_true)}")
            continue

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)

        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = None

        acc = accuracy_score(y_true, (y_pred >= 0.5).astype(int))

        result = {
            'name': name,
            'source': source,
            'n_total': len(prop_df),
            'n_evaluated': len(y_true),
            'n_pos': int(y_true.sum()),
            'n_neg': len(y_true) - int(y_true.sum()),
            'n_tokens': len(tokens),
            'accuracy': acc,
            'auc': auc
        }
        results.append(result)

        auc_str = f"{auc:.3f}" if auc else "N/A"
        print(f"  N={len(y_true)}, Pos={result['n_pos']}, Neg={result['n_neg']}, Acc={acc:.3f}, AUC={auc_str}")

    # Summary
    print("\n" + "=" * 90)
    print("SUMMARY")
    print("=" * 90)

    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values('auc', ascending=False)

        print(f"\n{'Endpoint':<40} {'N':>5} {'Pos':>5} {'Neg':>5} {'AUC':>7}")
        print("-" * 70)
        for _, row in results_df.iterrows():
            auc_str = f"{row['auc']:.3f}" if pd.notna(row['auc']) else "N/A"
            print(f"{row['name']:<40} {row['n_evaluated']:>5} {row['n_pos']:>5} {row['n_neg']:>5} {auc_str:>7}")

        aucs = results_df['auc'].dropna()
        print(f"\nTotal endpoints: {len(results_df)}")
        print(f"Mean AUC: {aucs.mean():.3f}")
        print(f"Median AUC: {aucs.median():.3f}")
        print(f"AUC >= 0.70: {len(aucs[aucs >= 0.70])}")
        print(f"AUC >= 0.65: {len(aucs[aucs >= 0.65])}")

        results_df.to_csv("publication/benchmark/results/semantic_eval/focused_eval_results.csv", index=False)
        print(f"\nSaved to: publication/benchmark/results/semantic_eval/focused_eval_results.csv")


if __name__ == "__main__":
    run_focused_eval()
