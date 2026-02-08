#!/usr/bin/env python3
"""
Quick semantic evaluation - samples compounds for faster testing.
"""

import pandas as pd
import numpy as np
import requests
import json
import time
import sys
import os
import re
import fnmatch
from urllib.parse import quote
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
import warnings
warnings.filterwarnings('ignore')

# Configuration
API_URL = os.environ.get("TOXTRANSFORMER_API", "http://136.111.102.10:6515")
DATA_DIR = "publication/benchmark/data"
RESULTS_DIR = "publication/benchmark/results/semantic_eval"
MAPPINGS_FILE = "scripts/curated_property_mappings.json"
MAX_SAMPLES_PER_PROPERTY = 100  # Sample for speed
MAX_WORKERS = 8
TIMEOUT = 120
RETRY_ATTEMPTS = 2

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_curated_mappings():
    """Load curated semantic property mappings."""
    with open(MAPPINGS_FILE) as f:
        mappings = json.load(f)

    endpoint_map = {}
    for category, category_data in mappings.items():
        if category.startswith("_") or category == "unmappable_endpoints":
            continue
        for endpoint_name, endpoint_data in category_data.items():
            if endpoint_name.startswith("_") or endpoint_data.get("skip_evaluation"):
                continue
            tokens = endpoint_data.get("tokens", [])
            if not tokens:
                continue
            patterns = endpoint_data.get("external_datasets", [])
            for pattern in patterns:
                endpoint_map[pattern] = {
                    "name": endpoint_name,
                    "category": category,
                    "tokens": tokens,
                    "description": endpoint_data.get("description", ""),
                }
    return endpoint_map, mappings


# Keyword matching rules - (required_keywords, excluded_keywords)
# Use more specific patterns to avoid false positives
KEYWORD_RULES = {
    "cyp1a2": (["cyp1a2"], ["cyp2", "cyp3"]),
    "cyp2c9": (["cyp2c9"], ["cyp1", "cyp2c19", "cyp2d", "cyp3"]),
    "cyp2c19": (["cyp2c19"], ["cyp1", "cyp2c9", "cyp2d", "cyp3"]),
    "cyp2d6": (["cyp2d6"], ["cyp1", "cyp2c", "cyp3"]),
    "cyp3a4": (["cyp3a4"], ["cyp1", "cyp2"]),
    "ames_mutagenicity": (["ames", "mutagen"], ["cytotox", "viability"]),
    "herg_inhibition": (["herg", "kcnh2"], []),
    "carcinogenicity": (["carcino"], ["mutagen"]),
    "aqueous_solubility": (["solub"], []),
    # Fixed: exclude proliferation, keratin, and require more specific ER patterns
    "estrogen_receptor_agonist": (["tox21_er", "era_", "estrogen_agonist", "nr_er"], ["antagon", "inhibit", "prolifer", "keratin"]),
    "estrogen_receptor_antagonist": (["tox21_er", "era_", "estrogen_antag"], ["agonist", "prolifer", "keratin"]),
    # Fixed: exclude _dn patterns for agonist, require agonist patterns
    "androgen_receptor_agonist": (["ar_bla_agonist", "ar_luc_agonist", "ar_mda_agonist"], ["_dn", "antagon"]),
    "androgen_receptor_antagonist": (["ar_bla_antagon", "ar_luc_antagon", "_ar_antag"], ["agonist"]),
    "ppar_gamma_agonist": (["pparg", "ppar_gamma"], ["antagon", "delta"]),
    "aromatase_inhibition": (["aromatase", "cyp19a1"], []),
    "nrf2_are_activation": (["nrf2", "are_cis"], ["inhibit"]),
    "p53_activation": (["p53_bla"], []),
    "zebrafish_developmental": (["zebrafish", "zf_120", "zf_144"], []),
    # Fixed: require toxicity-related terms, exclude mode of action
    "aquatic_acute": (["lc50", "ec50", "noec", "toxicity"], ["moa", "mode"]),
    # Fixed: require dili or hepatotoxicity specifically
    "hepatotoxicity": (["dili", "hepatotox"], ["clearance", "cyp"]),
}


def match_external_property(property_name, source, endpoint_map):
    """Find semantic match for an external property."""
    full_path = f"{source}/{property_name}"
    prop_lower = property_name.lower()

    # Pattern match
    for pattern, mapping in endpoint_map.items():
        if fnmatch.fnmatch(full_path, pattern) or fnmatch.fnmatch(property_name, pattern):
            return {
                "tokens": mapping["tokens"],
                "match_quality": "exact_pattern",
                "endpoint_name": mapping["name"],
                "category": mapping["category"],
            }

    # Keyword match
    for pattern, mapping in endpoint_map.items():
        endpoint_name = mapping["name"]
        if endpoint_name not in KEYWORD_RULES:
            continue
        required, excluded = KEYWORD_RULES[endpoint_name]
        if any(kw in prop_lower for kw in required):
            if not any(kw in prop_lower for kw in excluded):
                return {
                    "tokens": mapping["tokens"],
                    "match_quality": "keyword_match",
                    "endpoint_name": endpoint_name,
                    "category": mapping["category"],
                }
    return None


def predict_all_with_retry(inchi, max_retries=RETRY_ATTEMPTS):
    """Get all predictions for a compound."""
    for attempt in range(max_retries):
        try:
            url = f"{API_URL}/predict_all?inchi={quote(inchi, safe='')}"
            response = requests.get(url, timeout=TIMEOUT)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                time.sleep(2 ** attempt)
        except:
            if attempt < max_retries - 1:
                time.sleep(1)
    return None


def batch_predict(inchis):
    """Get predictions for a batch of compounds."""
    results = {}
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(predict_all_with_retry, inchi): inchi for inchi in inchis}
        for i, future in enumerate(as_completed(futures)):
            inchi = futures[future]
            try:
                preds = future.result()
                if preds is not None:
                    results[inchi] = preds
            except:
                pass
            if (i + 1) % 50 == 0:
                print(f"  {i+1}/{len(inchis)} compounds processed")
    return results


def extract_prediction_for_tokens(predictions, tokens):
    """Extract and average predictions for specific property tokens."""
    if predictions is None:
        return None
    values = []
    for pred in predictions:
        if pred.get('property_token') in tokens:
            val = pred.get('value')
            if val is not None and not np.isnan(val):
                values.append(val)
    if not values:
        return None
    return np.mean(values)


def run_quick_evaluation():
    """Run quick sampled evaluation."""
    print("=" * 70)
    print("ToxTransformer QUICK Semantic Evaluation")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print(f"Max samples per property: {MAX_SAMPLES_PER_PROPERTY}")

    # Load mappings
    print("\nLoading curated mappings...")
    endpoint_map, _ = load_curated_mappings()
    print(f"  Loaded {len(endpoint_map)} endpoint patterns")

    # Check API
    print("\nChecking API...")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        if resp.status_code != 200:
            print(f"API error: {resp.status_code}")
            sys.exit(1)
        print(f"  API OK")
    except Exception as e:
        print(f"API unreachable: {e}")
        sys.exit(1)

    # Load data
    print("\nLoading benchmark data...")
    binary_df = pd.read_parquet(f"{DATA_DIR}/external_binary_benchmark.parquet")
    print(f"  Binary: {len(binary_df):,} records")

    # Match properties
    print("\nMatching properties...")
    matched_props = []
    for source in binary_df['source'].unique():
        for prop in binary_df[binary_df['source'] == source]['property'].unique():
            match = match_external_property(prop, source, endpoint_map)
            if match:
                matched_props.append({
                    'source': source,
                    'property': prop,
                    **match
                })

    print(f"  Matched {len(matched_props)} binary properties")

    # Group by endpoint for evaluation
    by_endpoint = {}
    for m in matched_props:
        ep = m['endpoint_name']
        if ep not in by_endpoint:
            by_endpoint[ep] = []
        by_endpoint[ep].append(m)

    print(f"\n{len(by_endpoint)} unique endpoints to evaluate")

    # Collect samples
    print("\nCollecting samples...")
    all_samples = []
    for endpoint, props in by_endpoint.items():
        endpoint_samples = []
        for p in props:
            prop_df = binary_df[(binary_df['source'] == p['source']) &
                               (binary_df['property'] == p['property'])]
            # Balance classes
            pos = prop_df[prop_df['value'] == 1]
            neg = prop_df[prop_df['value'] == 0]
            n_per_class = min(MAX_SAMPLES_PER_PROPERTY // 2, len(pos), len(neg))
            if n_per_class < 5:
                continue
            samples = pd.concat([
                pos.sample(n=n_per_class, random_state=42),
                neg.sample(n=n_per_class, random_state=42)
            ])
            samples = samples.copy()
            samples['endpoint'] = endpoint
            samples['tokens'] = [p['tokens']] * len(samples)
            endpoint_samples.append(samples)

        if endpoint_samples:
            combined = pd.concat(endpoint_samples)
            # Limit total per endpoint
            if len(combined) > MAX_SAMPLES_PER_PROPERTY * 2:
                combined = combined.sample(n=MAX_SAMPLES_PER_PROPERTY * 2, random_state=42)
            all_samples.append(combined)

    if not all_samples:
        print("No samples to evaluate!")
        return

    samples_df = pd.concat(all_samples)
    unique_inchis = samples_df['inchi'].unique()
    print(f"  Total samples: {len(samples_df)}")
    print(f"  Unique compounds: {len(unique_inchis)}")

    # Get predictions
    print(f"\nFetching predictions...")
    start_time = time.time()
    predictions_cache = batch_predict(unique_inchis)
    elapsed = time.time() - start_time
    print(f"  Got {len(predictions_cache)} predictions in {elapsed:.1f}s")

    # Evaluate each endpoint
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    results = []
    for endpoint in sorted(by_endpoint.keys()):
        ep_df = samples_df[samples_df['endpoint'] == endpoint]
        if len(ep_df) < 10:
            continue

        y_true = []
        y_pred = []
        tokens = ep_df['tokens'].iloc[0]

        for _, row in ep_df.iterrows():
            if row['inchi'] not in predictions_cache:
                continue
            pred_val = extract_prediction_for_tokens(predictions_cache[row['inchi']], tokens)
            if pred_val is None:
                continue
            y_true.append(int(row['value']))
            y_pred.append(pred_val)

        if len(y_true) < 10 or len(set(y_true)) < 2:
            continue

        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        y_pred_binary = (y_pred >= 0.5).astype(int)

        try:
            auc = roc_auc_score(y_true, y_pred)
        except:
            auc = None

        acc = accuracy_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary, zero_division=0)

        result = {
            'endpoint': endpoint,
            'n_samples': len(y_true),
            'n_tokens': len(tokens),
            'accuracy': acc,
            'f1': f1,
            'auc': auc
        }
        results.append(result)

        auc_str = f"{auc:.3f}" if auc else "N/A"
        print(f"{endpoint:40s} N={len(y_true):4d}  Acc={acc:.3f}  F1={f1:.3f}  AUC={auc_str}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df.to_csv(f"{RESULTS_DIR}/quick_eval_results.csv", index=False)

        aucs = results_df['auc'].dropna()
        good_aucs = aucs[aucs >= 0.65]

        print(f"Total endpoints evaluated: {len(results_df)}")
        print(f"Mean AUC: {aucs.mean():.3f}")
        print(f"Median AUC: {aucs.median():.3f}")
        print(f"Endpoints with AUC >= 0.65: {len(good_aucs)}")

        print("\n--- Top performers (AUC >= 0.65) ---")
        top = results_df[results_df['auc'] >= 0.65].sort_values('auc', ascending=False)
        for _, row in top.iterrows():
            print(f"  {row['endpoint']:40s} AUC={row['auc']:.3f} (N={row['n_samples']})")

        print(f"\nResults saved to: {RESULTS_DIR}/quick_eval_results.csv")

    print(f"\nCompleted: {datetime.now().isoformat()}")
    return results_df


if __name__ == "__main__":
    run_quick_evaluation()
