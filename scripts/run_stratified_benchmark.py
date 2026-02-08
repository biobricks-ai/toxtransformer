#!/usr/bin/env python3
"""
Stratified External Benchmark Evaluation for ToxTransformer Publication

Uses stratified sampling to evaluate representative subsets efficiently.
Focuses on properties with clear mappings in the model.
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
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Configuration
API_URL = "http://136.111.102.10:6515"
DATA_DIR = "publication/benchmark/data"
RESULTS_DIR = "publication/benchmark/results"
MAX_WORKERS = 15  # Higher parallelism
TIMEOUT = 120
MAX_SAMPLES_PER_SOURCE = 200  # Stratified sampling limit (reduced for faster evaluation)
RETRY_ATTEMPTS = 2

os.makedirs(RESULTS_DIR, exist_ok=True)

def load_benchmark_data():
    """Load benchmark data with stratified sampling."""
    print("Loading benchmark data...")

    binary_df = pd.read_parquet(f"{DATA_DIR}/external_binary_benchmark.parquet")
    continuous_df = pd.read_parquet(f"{DATA_DIR}/external_continuous_benchmark.parquet")

    with open(f"{DATA_DIR}/benchmark_metadata.json") as f:
        metadata = json.load(f)

    print(f"  Full Binary: {len(binary_df):,} records, {binary_df['inchi'].nunique():,} compounds")
    print(f"  Full Continuous: {len(continuous_df):,} records, {continuous_df['inchi'].nunique():,} compounds")

    # Stratified sampling
    print(f"\nApplying stratified sampling (max {MAX_SAMPLES_PER_SOURCE} per source)...")

    binary_sampled = []
    for source in binary_df['source'].unique():
        source_df = binary_df[binary_df['source'] == source]
        unique_inchis = source_df['inchi'].unique()
        if len(unique_inchis) > MAX_SAMPLES_PER_SOURCE:
            np.random.seed(42)
            sampled_inchis = np.random.choice(unique_inchis, MAX_SAMPLES_PER_SOURCE, replace=False)
            source_df = source_df[source_df['inchi'].isin(sampled_inchis)]
        binary_sampled.append(source_df)
    binary_df = pd.concat(binary_sampled, ignore_index=True)

    continuous_sampled = []
    for source in continuous_df['source'].unique():
        source_df = continuous_df[continuous_df['source'] == source]
        unique_inchis = source_df['inchi'].unique()
        if len(unique_inchis) > MAX_SAMPLES_PER_SOURCE:
            np.random.seed(42)
            sampled_inchis = np.random.choice(unique_inchis, MAX_SAMPLES_PER_SOURCE, replace=False)
            source_df = source_df[source_df['inchi'].isin(sampled_inchis)]
        continuous_sampled.append(source_df)
    continuous_df = pd.concat(continuous_sampled, ignore_index=True)

    print(f"  Sampled Binary: {len(binary_df):,} records, {binary_df['inchi'].nunique():,} compounds")
    print(f"  Sampled Continuous: {len(continuous_df):,} records, {continuous_df['inchi'].nunique():,} compounds")

    return binary_df, continuous_df, metadata

def predict_with_retry(inchi, max_retries=RETRY_ATTEMPTS):
    """Get predictions for a compound with retry logic."""
    for attempt in range(max_retries):
        try:
            url = f"{API_URL}/predict_all?inchi={quote(inchi, safe='')}"
            response = requests.get(url, timeout=TIMEOUT)
            if response.status_code == 200:
                return inchi, response.json()
            elif response.status_code == 429:
                time.sleep(2 ** attempt)
        except:
            if attempt < max_retries - 1:
                time.sleep(0.5)
    return inchi, None

def batch_predict(inchis, progress_callback=None):
    """Get predictions for compounds with high parallelism."""
    results = {}
    failed = []
    total = len(inchis)

    print(f"  Processing {total:,} compounds with {MAX_WORKERS} parallel workers...")

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(predict_with_retry, inchi): inchi for inchi in inchis}

        for i, future in enumerate(as_completed(futures)):
            inchi, preds = future.result()
            if preds is not None:
                results[inchi] = preds
            else:
                failed.append(inchi)

            if (i + 1) % 50 == 0 or (i + 1) == total:
                elapsed = time.time() - start_time
                rate = (i + 1) / elapsed * 60
                eta = (total - i - 1) / rate if rate > 0 else 0
                print(f"  [{i+1:,}/{total:,}] {len(results):,} OK, {len(failed)} failed | {rate:.1f}/min | ETA: {eta:.1f} min")

    return results, failed

def extract_prediction(predictions, property_tokens):
    """Extract and average predictions for matching tokens."""
    if predictions is None:
        return None

    values = []
    for pred in predictions:
        if pred.get('property_token') in property_tokens:
            val = pred.get('value')
            if val is not None and not np.isnan(val):
                values.append(val)

    return np.mean(values) if values else None

def evaluate_binary_source(source_df, predictions_cache, property_mapping):
    """Evaluate binary classification for a source."""
    results = []

    for prop in source_df['property'].unique():
        # Find matching property tokens
        prop_tokens = []
        for key, mappings in property_mapping.items():
            if prop in key or any(kw in prop.lower() for kw in ['ames', 'carc', 'muta', 'herg', 'cyp', 'dili', 'tox']):
                prop_tokens.extend([m[0] for m in mappings])

        if not prop_tokens:
            continue

        prop_df = source_df[source_df['property'] == prop]
        prop_results = []

        for _, row in prop_df.iterrows():
            if row['inchi'] not in predictions_cache:
                continue

            pred_val = extract_prediction(predictions_cache[row['inchi']], prop_tokens)
            if pred_val is None:
                continue

            prop_results.append({
                'inchi': row['inchi'],
                'expected': int(row['value']),
                'predicted_prob': pred_val,
                'predicted_binary': 1 if pred_val >= 0.5 else 0,
                'property': prop
            })

        if len(prop_results) < 10:
            continue

        pdf = pd.DataFrame(prop_results)
        y_true = pdf['expected'].values
        y_pred_prob = pdf['predicted_prob'].values
        y_pred = pdf['predicted_binary'].values

        metrics = {
            'property': prop,
            'n_samples': len(prop_results),
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }

        if len(np.unique(y_true)) > 1:
            try:
                metrics['auc'] = roc_auc_score(y_true, y_pred_prob)
            except:
                metrics['auc'] = None
        else:
            metrics['auc'] = None

        results.append(metrics)

    return results

def evaluate_continuous_source(source_df, predictions_cache, property_mapping):
    """Evaluate regression for a source."""
    results = []

    for prop in source_df['property'].unique():
        # Find matching property tokens
        prop_tokens = []
        for key, mappings in property_mapping.items():
            if prop in key or any(kw in prop.lower() for kw in ['solubility', 'clearance', 'half', 'ld50', 'ec50', 'ic50']):
                prop_tokens.extend([m[0] for m in mappings])

        if not prop_tokens:
            continue

        prop_df = source_df[source_df['property'] == prop]
        prop_results = []

        for _, row in prop_df.iterrows():
            if row['inchi'] not in predictions_cache:
                continue

            pred_val = extract_prediction(predictions_cache[row['inchi']], prop_tokens)
            if pred_val is None:
                continue

            prop_results.append({
                'inchi': row['inchi'],
                'expected': float(row['value']),
                'predicted': pred_val,
                'property': prop
            })

        if len(prop_results) < 10:
            continue

        pdf = pd.DataFrame(prop_results)
        y_true = pdf['expected'].values
        y_pred = pdf['predicted'].values

        metrics = {
            'property': prop,
            'n_samples': len(prop_results),
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
        }

        try:
            metrics['pearson_r'], _ = pearsonr(y_true, y_pred)
            metrics['spearman_r'], _ = spearmanr(y_true, y_pred)
        except:
            metrics['pearson_r'] = metrics['spearman_r'] = None

        results.append(metrics)

    return results

def run_evaluation():
    """Run stratified benchmark evaluation."""
    global start_time

    print("=" * 70)
    print("ToxTransformer Stratified Benchmark Evaluation")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")

    # Check API
    print("\nChecking API health...")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        if resp.status_code != 200:
            print(f"ERROR: API unhealthy")
            sys.exit(1)
        print(f"  API healthy")
    except Exception as e:
        print(f"ERROR: API unreachable: {e}")
        sys.exit(1)

    # Load data
    binary_df, continuous_df, metadata = load_benchmark_data()
    property_mapping = metadata.get('property_mapping', {})

    # Get unique compounds
    all_inchis = list(set(binary_df['inchi'].unique()) | set(continuous_df['inchi'].unique()))
    print(f"\nTotal unique compounds to evaluate: {len(all_inchis):,}")

    # Fetch predictions
    print("\nFetching predictions...")
    start_time = time.time()
    predictions_cache, failed = batch_predict(all_inchis)

    elapsed = time.time() - start_time
    print(f"\nPredictions complete in {elapsed/60:.1f} minutes")
    print(f"  Successful: {len(predictions_cache):,}")
    print(f"  Failed: {len(failed)}")

    # Save cache
    cache_file = f"{RESULTS_DIR}/predictions_cache.json"
    with open(cache_file, 'w') as f:
        json.dump(predictions_cache, f)
    print(f"  Saved to: {cache_file}")

    # Evaluate binary
    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION RESULTS")
    print("=" * 70)

    all_binary_results = []
    for source in binary_df['source'].unique():
        source_df = binary_df[binary_df['source'] == source]
        results = evaluate_binary_source(source_df, predictions_cache, property_mapping)

        if results:
            print(f"\n{source}:")
            for r in results:
                r['source'] = source
                all_binary_results.append(r)
                auc_str = f", AUC={r['auc']:.3f}" if r.get('auc') else ""
                print(f"  {r['property'][:50]}: N={r['n_samples']}, Acc={r['accuracy']:.3f}, F1={r['f1']:.3f}{auc_str}")

    if all_binary_results:
        binary_summary = pd.DataFrame(all_binary_results)
        binary_summary.to_csv(f"{RESULTS_DIR}/binary_classification_summary.csv", index=False)

        print(f"\n--- Binary Summary ---")
        print(f"Evaluations: {len(all_binary_results)}")
        print(f"Total samples: {binary_summary['n_samples'].sum():,}")
        print(f"Mean Accuracy: {binary_summary['accuracy'].mean():.3f}")
        print(f"Mean F1: {binary_summary['f1'].mean():.3f}")
        auc_vals = binary_summary['auc'].dropna()
        if len(auc_vals) > 0:
            print(f"Mean AUC: {auc_vals.mean():.3f}")

    # Evaluate continuous
    print("\n" + "=" * 70)
    print("CONTINUOUS REGRESSION RESULTS")
    print("=" * 70)

    all_continuous_results = []
    for source in continuous_df['source'].unique():
        source_df = continuous_df[continuous_df['source'] == source]
        results = evaluate_continuous_source(source_df, predictions_cache, property_mapping)

        if results:
            print(f"\n{source}:")
            for r in results:
                r['source'] = source
                all_continuous_results.append(r)
                r_str = f", r={r['pearson_r']:.3f}" if r.get('pearson_r') else ""
                print(f"  {r['property'][:50]}: N={r['n_samples']}, RMSE={r['rmse']:.3f}, R2={r['r2']:.3f}{r_str}")

    if all_continuous_results:
        continuous_summary = pd.DataFrame(all_continuous_results)
        continuous_summary.to_csv(f"{RESULTS_DIR}/continuous_regression_summary.csv", index=False)

        print(f"\n--- Continuous Summary ---")
        print(f"Evaluations: {len(all_continuous_results)}")
        print(f"Total samples: {continuous_summary['n_samples'].sum():,}")
        print(f"Mean RMSE: {continuous_summary['rmse'].mean():.3f}")
        print(f"Mean R2: {continuous_summary['r2'].mean():.3f}")
        r_vals = continuous_summary['pearson_r'].dropna()
        if len(r_vals) > 0:
            print(f"Mean Pearson r: {r_vals.mean():.3f}")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_compounds_evaluated': len(predictions_cache),
        'total_compounds_failed': len(failed),
        'evaluation_time_minutes': elapsed / 60,
        'binary_classification': {
            'n_evaluations': len(all_binary_results),
            'total_samples': int(binary_summary['n_samples'].sum()) if all_binary_results else 0,
            'mean_accuracy': float(binary_summary['accuracy'].mean()) if all_binary_results else None,
            'mean_f1': float(binary_summary['f1'].mean()) if all_binary_results else None,
            'mean_auc': float(auc_vals.mean()) if len(auc_vals) > 0 else None,
        },
        'continuous_regression': {
            'n_evaluations': len(all_continuous_results),
            'total_samples': int(continuous_summary['n_samples'].sum()) if all_continuous_results else 0,
            'mean_rmse': float(continuous_summary['rmse'].mean()) if all_continuous_results else None,
            'mean_r2': float(continuous_summary['r2'].mean()) if all_continuous_results else None,
            'mean_pearson_r': float(r_vals.mean()) if len(r_vals) > 0 else None,
        }
    }

    with open(f"{RESULTS_DIR}/evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Completed: {datetime.now().isoformat()}")

    return summary

if __name__ == "__main__":
    run_evaluation()
