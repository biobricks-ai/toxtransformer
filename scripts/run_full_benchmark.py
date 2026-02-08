#!/usr/bin/env python3
"""
Full External Benchmark Evaluation for ToxTransformer Publication

Evaluates ToxTransformer predictions against external benchmark datasets:
- Binary classification: 93,475 records from 15 sources
- Continuous regression: 147,119 records from 18 sources

Uses batch API calls with progress tracking and comprehensive metrics.
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
BATCH_SIZE = 50  # Compounds per batch
MAX_WORKERS = 4  # Concurrent API calls
TIMEOUT = 180  # seconds per request
RETRY_ATTEMPTS = 3

# Ensure results directory exists
os.makedirs(RESULTS_DIR, exist_ok=True)

def load_benchmark_data():
    """Load benchmark data and metadata."""
    print("Loading benchmark data...")

    binary_df = pd.read_parquet(f"{DATA_DIR}/external_binary_benchmark.parquet")
    continuous_df = pd.read_parquet(f"{DATA_DIR}/external_continuous_benchmark.parquet")

    with open(f"{DATA_DIR}/benchmark_metadata.json") as f:
        metadata = json.load(f)

    print(f"  Binary: {len(binary_df):,} records, {binary_df['inchi'].nunique():,} compounds")
    print(f"  Continuous: {len(continuous_df):,} records, {continuous_df['inchi'].nunique():,} compounds")

    return binary_df, continuous_df, metadata

def predict_all_with_retry(inchi, max_retries=RETRY_ATTEMPTS):
    """Get all predictions for a compound with retry logic."""
    for attempt in range(max_retries):
        try:
            url = f"{API_URL}/predict_all?inchi={quote(inchi, safe='')}"
            response = requests.get(url, timeout=TIMEOUT)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:  # Rate limited
                time.sleep(2 ** attempt)
                continue
        except requests.exceptions.Timeout:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(1)
                continue
    return None

def batch_predict(inchis, progress_prefix=""):
    """Get predictions for a batch of compounds with progress tracking."""
    results = {}
    failed = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(predict_all_with_retry, inchi): inchi for inchi in inchis}

        for i, future in enumerate(as_completed(futures)):
            inchi = futures[future]
            try:
                preds = future.result()
                if preds is not None:
                    results[inchi] = preds
                else:
                    failed.append(inchi)
            except Exception as e:
                failed.append(inchi)

            if (i + 1) % 100 == 0 or (i + 1) == len(inchis):
                print(f"  {progress_prefix}{i+1}/{len(inchis)} compounds processed, {len(results)} successful, {len(failed)} failed")

    return results, failed

def extract_prediction_for_property(predictions, property_tokens):
    """Extract and average predictions for matching property tokens."""
    if predictions is None:
        return None

    pred_values = []
    for pred in predictions:
        if pred.get('property_token') in property_tokens:
            val = pred.get('value')
            if val is not None and not np.isnan(val):
                pred_values.append(val)

    if not pred_values:
        return None

    return np.mean(pred_values)

def evaluate_binary_classification(df, predictions_cache, property_mapping, property_name):
    """Evaluate binary classification for a specific property."""
    results = []

    # Get property tokens from mapping
    mapping_key = None
    for key in property_mapping:
        if property_name in key:
            mapping_key = key
            break

    if mapping_key is None:
        return None, None

    property_tokens = [m[0] for m in property_mapping[mapping_key]]

    for _, row in df.iterrows():
        inchi = row['inchi']
        if inchi not in predictions_cache:
            continue

        pred_value = extract_prediction_for_property(predictions_cache[inchi], property_tokens)
        if pred_value is None:
            continue

        results.append({
            'inchi': inchi,
            'expected': int(row['value']),
            'predicted_prob': pred_value,
            'predicted_binary': 1 if pred_value >= 0.5 else 0,
            'property': row['property'],
            'source': row['source']
        })

    if len(results) < 10:
        return None, None

    results_df = pd.DataFrame(results)
    y_true = results_df['expected'].values
    y_pred_prob = results_df['predicted_prob'].values
    y_pred_binary = results_df['predicted_binary'].values

    metrics = {
        'n_samples': len(results),
        'accuracy': accuracy_score(y_true, y_pred_binary),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
    }

    # AUC requires both classes present
    if len(np.unique(y_true)) > 1:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_prob)
        except:
            metrics['auc'] = None
    else:
        metrics['auc'] = None

    return metrics, results_df

def evaluate_regression(df, predictions_cache, property_mapping, property_name):
    """Evaluate regression for a specific property."""
    results = []

    # Get property tokens from mapping
    mapping_key = None
    for key in property_mapping:
        if property_name in key:
            mapping_key = key
            break

    if mapping_key is None:
        return None, None

    property_tokens = [m[0] for m in property_mapping[mapping_key]]

    for _, row in df.iterrows():
        inchi = row['inchi']
        if inchi not in predictions_cache:
            continue

        pred_value = extract_prediction_for_property(predictions_cache[inchi], property_tokens)
        if pred_value is None:
            continue

        results.append({
            'inchi': inchi,
            'expected': float(row['value']),
            'predicted': pred_value,
            'property': row['property'],
            'source': row['source']
        })

    if len(results) < 10:
        return None, None

    results_df = pd.DataFrame(results)
    y_true = results_df['expected'].values
    y_pred = results_df['predicted'].values

    metrics = {
        'n_samples': len(results),
        'mse': mean_squared_error(y_true, y_pred),
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
    }

    # Correlation metrics
    try:
        metrics['pearson_r'], metrics['pearson_p'] = pearsonr(y_true, y_pred)
        metrics['spearman_r'], metrics['spearman_p'] = spearmanr(y_true, y_pred)
    except:
        metrics['pearson_r'] = metrics['pearson_p'] = None
        metrics['spearman_r'] = metrics['spearman_p'] = None

    return metrics, results_df

def run_full_evaluation():
    """Run complete benchmark evaluation."""
    print("=" * 70)
    print("ToxTransformer External Benchmark Evaluation")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Check API health
    print("Checking API health...")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        if resp.status_code != 200:
            print(f"ERROR: API unhealthy (HTTP {resp.status_code})")
            sys.exit(1)
        print(f"  API healthy: {resp.json()}")
    except Exception as e:
        print(f"ERROR: API unreachable: {e}")
        sys.exit(1)

    # Load data
    binary_df, continuous_df, metadata = load_benchmark_data()
    property_mapping = metadata.get('property_mapping', {})

    # Get unique compounds
    all_inchis = set(binary_df['inchi'].unique()) | set(continuous_df['inchi'].unique())
    print(f"\nTotal unique compounds to evaluate: {len(all_inchis):,}")

    # Get predictions for all compounds
    print(f"\nFetching predictions from API...")
    start_time = time.time()

    predictions_cache, failed = batch_predict(list(all_inchis), progress_prefix="")

    elapsed = time.time() - start_time
    print(f"\nPredictions complete:")
    print(f"  Successful: {len(predictions_cache):,}")
    print(f"  Failed: {len(failed):,}")
    print(f"  Time: {elapsed/60:.1f} minutes")

    # Save predictions cache
    cache_file = f"{RESULTS_DIR}/predictions_cache.json"
    with open(cache_file, 'w') as f:
        json.dump(predictions_cache, f)
    print(f"  Saved to: {cache_file}")

    # Evaluate by source
    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION RESULTS")
    print("=" * 70)

    binary_results = []
    for source in binary_df['source'].unique():
        source_df = binary_df[binary_df['source'] == source]

        # Get unique properties in this source
        for prop in source_df['property'].unique():
            prop_df = source_df[source_df['property'] == prop]

            metrics, details_df = evaluate_binary_classification(
                prop_df, predictions_cache, property_mapping, prop
            )

            if metrics is not None:
                metrics['source'] = source
                metrics['property'] = prop
                binary_results.append(metrics)

                print(f"\n{source} / {prop}:")
                print(f"  N={metrics['n_samples']}, Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}", end="")
                if metrics.get('auc'):
                    print(f", AUC={metrics['auc']:.3f}")
                else:
                    print()

                # Save detailed results
                if details_df is not None:
                    safe_name = prop.replace('/', '_').replace(' ', '_')[:50]
                    details_df.to_csv(f"{RESULTS_DIR}/binary_{source}_{safe_name}_details.csv", index=False)

    # Save binary summary
    if binary_results:
        binary_summary = pd.DataFrame(binary_results)
        binary_summary.to_csv(f"{RESULTS_DIR}/binary_classification_summary.csv", index=False)

        print(f"\n--- Binary Classification Summary ---")
        print(f"Total evaluations: {len(binary_results)}")
        print(f"Mean Accuracy: {binary_summary['accuracy'].mean():.3f}")
        print(f"Mean F1: {binary_summary['f1'].mean():.3f}")
        auc_values = binary_summary['auc'].dropna()
        if len(auc_values) > 0:
            print(f"Mean AUC: {auc_values.mean():.3f}")

    # Evaluate continuous
    print("\n" + "=" * 70)
    print("CONTINUOUS REGRESSION RESULTS")
    print("=" * 70)

    continuous_results = []
    for source in continuous_df['source'].unique():
        source_df = continuous_df[continuous_df['source'] == source]

        for prop in source_df['property'].unique():
            prop_df = source_df[source_df['property'] == prop]

            metrics, details_df = evaluate_regression(
                prop_df, predictions_cache, property_mapping, prop
            )

            if metrics is not None:
                metrics['source'] = source
                metrics['property'] = prop
                continuous_results.append(metrics)

                print(f"\n{source} / {prop}:")
                print(f"  N={metrics['n_samples']}, RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}", end="")
                if metrics.get('pearson_r') is not None:
                    print(f", r={metrics['pearson_r']:.3f}")
                else:
                    print()

                # Save detailed results
                if details_df is not None:
                    safe_name = prop.replace('/', '_').replace(' ', '_')[:50]
                    details_df.to_csv(f"{RESULTS_DIR}/continuous_{source}_{safe_name}_details.csv", index=False)

    # Save continuous summary
    if continuous_results:
        continuous_summary = pd.DataFrame(continuous_results)
        continuous_summary.to_csv(f"{RESULTS_DIR}/continuous_regression_summary.csv", index=False)

        print(f"\n--- Continuous Regression Summary ---")
        print(f"Total evaluations: {len(continuous_results)}")
        print(f"Mean RMSE: {continuous_summary['rmse'].mean():.3f}")
        print(f"Mean R2: {continuous_summary['r2'].mean():.3f}")
        pearson_values = continuous_summary['pearson_r'].dropna()
        if len(pearson_values) > 0:
            print(f"Mean Pearson r: {pearson_values.mean():.3f}")

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_compounds_evaluated': len(predictions_cache),
        'total_compounds_failed': len(failed),
        'binary_classification': {
            'n_evaluations': len(binary_results),
            'mean_accuracy': binary_summary['accuracy'].mean() if binary_results else None,
            'mean_f1': binary_summary['f1'].mean() if binary_results else None,
            'mean_auc': auc_values.mean() if len(auc_values) > 0 else None,
        },
        'continuous_regression': {
            'n_evaluations': len(continuous_results),
            'mean_rmse': continuous_summary['rmse'].mean() if continuous_results else None,
            'mean_r2': continuous_summary['r2'].mean() if continuous_results else None,
            'mean_pearson_r': pearson_values.mean() if len(pearson_values) > 0 else None,
        }
    }

    with open(f"{RESULTS_DIR}/evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Completed: {datetime.now().isoformat()}")

    return summary

if __name__ == "__main__":
    run_full_evaluation()
