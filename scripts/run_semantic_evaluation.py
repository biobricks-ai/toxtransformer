#!/usr/bin/env python3
"""
Semantic External Evaluation for ToxTransformer Publication

This script performs external evaluation using CURATED semantic property mappings
to ensure we only evaluate endpoints where ToxTransformer has semantically
appropriate training data.

Key differences from run_full_benchmark.py:
1. Uses curated_property_mappings.json for strict semantic matching
2. Only evaluates endpoints with verified semantic equivalence
3. Reports match quality alongside performance metrics
4. Skips endpoints without appropriate tokens rather than using proxies
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
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score, precision_score, recall_score,
    mean_squared_error, mean_absolute_error, r2_score
)
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Configuration
API_URL = os.environ.get("TOXTRANSFORMER_API", "http://136.111.102.10:6515")
DATA_DIR = "publication/benchmark/data"
RESULTS_DIR = "publication/benchmark/results/semantic_eval"
MAPPINGS_FILE = "scripts/curated_property_mappings.json"
BATCH_SIZE = 50
MAX_WORKERS = 4
TIMEOUT = 180
RETRY_ATTEMPTS = 3

os.makedirs(RESULTS_DIR, exist_ok=True)


def load_curated_mappings():
    """Load curated semantic property mappings."""
    with open(MAPPINGS_FILE) as f:
        mappings = json.load(f)

    # Flatten into endpoint -> tokens mapping
    endpoint_map = {}

    for category, category_data in mappings.items():
        if category.startswith("_"):
            continue
        if category == "unmappable_endpoints":
            continue

        for endpoint_name, endpoint_data in category_data.items():
            if endpoint_name.startswith("_"):
                continue
            if endpoint_data.get("skip_evaluation"):
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
                    "excluded_patterns": endpoint_data.get("excluded_patterns", [])
                }

    return endpoint_map, mappings


def match_external_property(property_name, source, endpoint_map):
    """
    Find semantic match for an external property.

    Returns:
        dict with 'tokens', 'match_quality', 'matched_pattern' or None if no match
    """
    # Create full property path for matching
    full_path = f"{source}/{property_name}"
    prop_lower = property_name.lower()

    for pattern, mapping in endpoint_map.items():
        # Try exact pattern match first
        if fnmatch.fnmatch(full_path, pattern) or fnmatch.fnmatch(property_name, pattern):
            return {
                "tokens": mapping["tokens"],
                "match_quality": "exact_pattern",
                "matched_pattern": pattern,
                "endpoint_name": mapping["name"],
                "category": mapping["category"],
                "description": mapping["description"]
            }

    # Keyword matching rules: endpoint_name -> (required_keywords, excluded_keywords)
    # Use specific patterns to avoid false positives
    keyword_rules = {
        # CYP isoforms
        "cyp1a2": (["cyp1a2"], ["cyp2", "cyp3"]),
        "cyp2c9": (["cyp2c9"], ["cyp1", "cyp2c19", "cyp2d", "cyp3"]),
        "cyp2c19": (["cyp2c19"], ["cyp1", "cyp2c9", "cyp2d", "cyp3"]),
        "cyp2d6": (["cyp2d6"], ["cyp1", "cyp2c", "cyp3"]),
        "cyp3a4": (["cyp3a4"], ["cyp1", "cyp2"]),

        # Mutagenicity
        "ames_mutagenicity": (["ames", "mutagen"], ["cytotox", "viability"]),

        # Cardiac
        "herg_inhibition": (["herg", "kcnh2"], []),

        # Carcinogenicity
        "carcinogenicity": (["carcino"], ["mutagen"]),

        # Solubility
        "aqueous_solubility": (["solub"], []),

        # Nuclear receptors - fixed to avoid false positives
        "estrogen_receptor_agonist": (["tox21_er", "era_", "estrogen_agonist", "nr_er"], ["antagon", "inhibit", "prolifer", "keratin"]),
        "estrogen_receptor_antagonist": (["tox21_er", "era_", "estrogen_antag"], ["agonist", "prolifer", "keratin"]),
        "androgen_receptor_agonist": (["ar_bla_agonist", "ar_luc_agonist", "ar_mda_agonist"], ["_dn", "antagon"]),
        "androgen_receptor_antagonist": (["ar_bla_antagon", "ar_luc_antagon", "_ar_antag"], ["agonist"]),
        "ppar_gamma_agonist": (["pparg", "ppar_gamma"], ["antagon", "delta"]),
        "aromatase_inhibition": (["aromatase", "cyp19a1"], []),

        # Stress pathways
        "nrf2_are_activation": (["nrf2", "are_cis"], ["inhibit"]),
        "p53_activation": (["p53_bla"], []),

        # Aquatic/zebrafish
        "zebrafish_developmental": (["zebrafish", "zf_120", "zf_144"], []),
        "aquatic_acute": (["lc50", "ec50", "noec", "toxicity"], ["moa", "mode"]),

        # Hepatotoxicity - specific DILI only
        "hepatotoxicity": (["dili", "hepatotox"], ["clearance", "cyp"]),
    }

    # Try semantic keyword matching
    for pattern, mapping in endpoint_map.items():
        endpoint_name = mapping["name"]

        if endpoint_name not in keyword_rules:
            continue

        required_kws, excluded_kws = keyword_rules[endpoint_name]

        # Check if any required keyword is present
        has_required = any(kw in prop_lower for kw in required_kws)
        if not has_required:
            continue

        # Check if any excluded keyword is present
        has_excluded = any(kw in prop_lower for kw in excluded_kws)
        if has_excluded:
            continue

        return {
            "tokens": mapping["tokens"],
            "match_quality": "keyword_match",
            "matched_pattern": pattern,
            "endpoint_name": endpoint_name,
            "category": mapping["category"],
            "description": mapping["description"]
        }

    return None


def load_benchmark_data():
    """Load benchmark data and metadata."""
    print("Loading benchmark data...")

    binary_df = pd.read_parquet(f"{DATA_DIR}/external_binary_benchmark.parquet")
    continuous_df = pd.read_parquet(f"{DATA_DIR}/external_continuous_benchmark.parquet")

    print(f"  Binary: {len(binary_df):,} records, {binary_df['inchi'].nunique():,} compounds")
    print(f"  Continuous: {len(continuous_df):,} records, {continuous_df['inchi'].nunique():,} compounds")

    return binary_df, continuous_df


def predict_all_with_retry(inchi, max_retries=RETRY_ATTEMPTS):
    """Get all predictions for a compound with retry logic."""
    for attempt in range(max_retries):
        try:
            url = f"{API_URL}/predict_all?inchi={quote(inchi, safe='')}"
            response = requests.get(url, timeout=TIMEOUT)
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
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


def extract_prediction_for_tokens(predictions, tokens):
    """Extract and average predictions for specific property tokens."""
    if predictions is None:
        return None

    pred_values = []
    matched_tokens = []

    for pred in predictions:
        token = pred.get('property_token')
        if token in tokens:
            val = pred.get('value')
            if val is not None and not np.isnan(val):
                pred_values.append(val)
                matched_tokens.append(token)

    if not pred_values:
        return None

    return {
        "value": np.mean(pred_values),
        "n_tokens_matched": len(pred_values),
        "tokens_used": matched_tokens
    }


def evaluate_binary_classification(df, predictions_cache, match_info):
    """Evaluate binary classification for a matched property."""
    results = []
    tokens = match_info["tokens"]

    for _, row in df.iterrows():
        inchi = row['inchi']
        if inchi not in predictions_cache:
            continue

        pred_result = extract_prediction_for_tokens(predictions_cache[inchi], tokens)
        if pred_result is None:
            continue

        results.append({
            'inchi': inchi,
            'expected': int(row['value']),
            'predicted_prob': pred_result["value"],
            'predicted_binary': 1 if pred_result["value"] >= 0.5 else 0,
            'n_tokens': pred_result["n_tokens_matched"],
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
        'match_quality': match_info["match_quality"],
        'endpoint_name': match_info["endpoint_name"],
        'category': match_info["category"],
        'n_tokens_used': len(tokens),
        'tokens': tokens
    }

    if len(np.unique(y_true)) > 1:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_pred_prob)
        except:
            metrics['auc'] = None
    else:
        metrics['auc'] = None

    return metrics, results_df


def evaluate_regression(df, predictions_cache, match_info):
    """Evaluate regression for a matched property."""
    results = []
    tokens = match_info["tokens"]

    for _, row in df.iterrows():
        inchi = row['inchi']
        if inchi not in predictions_cache:
            continue

        pred_result = extract_prediction_for_tokens(predictions_cache[inchi], tokens)
        if pred_result is None:
            continue

        results.append({
            'inchi': inchi,
            'expected': float(row['value']),
            'predicted': pred_result["value"],
            'n_tokens': pred_result["n_tokens_matched"],
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
        'match_quality': match_info["match_quality"],
        'endpoint_name': match_info["endpoint_name"],
        'category': match_info["category"],
        'n_tokens_used': len(tokens),
        'tokens': tokens
    }

    try:
        metrics['pearson_r'], metrics['pearson_p'] = pearsonr(y_true, y_pred)
        metrics['spearman_r'], metrics['spearman_p'] = spearmanr(y_true, y_pred)
    except:
        metrics['pearson_r'] = metrics['pearson_p'] = None
        metrics['spearman_r'] = metrics['spearman_p'] = None

    return metrics, results_df


def run_semantic_evaluation():
    """Run semantic benchmark evaluation."""
    print("=" * 70)
    print("ToxTransformer SEMANTIC External Benchmark Evaluation")
    print("=" * 70)
    print(f"Started: {datetime.now().isoformat()}")
    print()

    # Load curated mappings
    print("Loading curated semantic mappings...")
    endpoint_map, full_mappings = load_curated_mappings()
    print(f"  Loaded {len(endpoint_map)} endpoint patterns")

    # Check API health
    print("\nChecking API health...")
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
    binary_df, continuous_df = load_benchmark_data()

    # Analyze which properties can be evaluated
    print("\n" + "=" * 70)
    print("PROPERTY MATCHING ANALYSIS")
    print("=" * 70)

    matched_properties = []
    unmatched_properties = []

    all_properties = set()
    for source in binary_df['source'].unique():
        for prop in binary_df[binary_df['source'] == source]['property'].unique():
            all_properties.add((source, prop, 'binary'))
    for source in continuous_df['source'].unique():
        for prop in continuous_df[continuous_df['source'] == source]['property'].unique():
            all_properties.add((source, prop, 'continuous'))

    for source, prop, data_type in all_properties:
        match = match_external_property(prop, source, endpoint_map)
        if match:
            matched_properties.append({
                'source': source,
                'property': prop,
                'data_type': data_type,
                'match': match
            })
        else:
            unmatched_properties.append({
                'source': source,
                'property': prop,
                'data_type': data_type
            })

    print(f"\nMatched properties: {len(matched_properties)}")
    print(f"Unmatched properties: {len(unmatched_properties)}")

    # Save matching report
    matching_report = {
        'matched': [
            {
                'source': m['source'],
                'property': m['property'],
                'data_type': m['data_type'],
                'endpoint_name': m['match']['endpoint_name'],
                'category': m['match']['category'],
                'match_quality': m['match']['match_quality'],
                'tokens': m['match']['tokens']
            }
            for m in matched_properties
        ],
        'unmatched': unmatched_properties
    }

    with open(f"{RESULTS_DIR}/property_matching_report.json", 'w') as f:
        json.dump(matching_report, f, indent=2)
    print(f"\nMatching report saved to: {RESULTS_DIR}/property_matching_report.json")

    # Get unique compounds for matched properties only
    matched_binary = [(m['source'], m['property']) for m in matched_properties if m['data_type'] == 'binary']
    matched_continuous = [(m['source'], m['property']) for m in matched_properties if m['data_type'] == 'continuous']

    # Filter dataframes
    binary_mask = binary_df.apply(lambda r: (r['source'], r['property']) in matched_binary, axis=1)
    continuous_mask = continuous_df.apply(lambda r: (r['source'], r['property']) in matched_continuous, axis=1)

    filtered_binary = binary_df[binary_mask]
    filtered_continuous = continuous_df[continuous_mask]

    all_inchis = set(filtered_binary['inchi'].unique()) | set(filtered_continuous['inchi'].unique())
    print(f"\nTotal unique compounds to evaluate (matched properties only): {len(all_inchis):,}")

    if len(all_inchis) == 0:
        print("No compounds to evaluate - check property matching!")
        return

    # Get predictions
    print(f"\nFetching predictions from API...")
    start_time = time.time()

    predictions_cache, failed = batch_predict(list(all_inchis))

    elapsed = time.time() - start_time
    print(f"\nPredictions complete:")
    print(f"  Successful: {len(predictions_cache):,}")
    print(f"  Failed: {len(failed):,}")
    print(f"  Time: {elapsed/60:.1f} minutes")

    # Evaluate binary classification
    print("\n" + "=" * 70)
    print("BINARY CLASSIFICATION RESULTS (Semantic Matches Only)")
    print("=" * 70)

    binary_results = []
    for m in matched_properties:
        if m['data_type'] != 'binary':
            continue

        source, prop = m['source'], m['property']
        prop_df = binary_df[(binary_df['source'] == source) & (binary_df['property'] == prop)]

        metrics, details_df = evaluate_binary_classification(
            prop_df, predictions_cache, m['match']
        )

        if metrics is not None:
            metrics['source'] = source
            metrics['property'] = prop
            binary_results.append(metrics)

            print(f"\n{source} / {prop}:")
            print(f"  Match: {metrics['endpoint_name']} ({metrics['match_quality']})")
            print(f"  N={metrics['n_samples']}, Acc={metrics['accuracy']:.3f}, F1={metrics['f1']:.3f}", end="")
            if metrics.get('auc'):
                print(f", AUC={metrics['auc']:.3f}")
            else:
                print()

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

        # Group by match quality
        print("\nResults by match quality:")
        for quality in binary_summary['match_quality'].unique():
            subset = binary_summary[binary_summary['match_quality'] == quality]
            aucs = subset['auc'].dropna()
            print(f"  {quality}: {len(subset)} evaluations, mean AUC={aucs.mean():.3f}" if len(aucs) > 0 else f"  {quality}: {len(subset)} evaluations")

    # Evaluate continuous regression
    print("\n" + "=" * 70)
    print("CONTINUOUS REGRESSION RESULTS (Semantic Matches Only)")
    print("=" * 70)

    continuous_results = []
    for m in matched_properties:
        if m['data_type'] != 'continuous':
            continue

        source, prop = m['source'], m['property']
        prop_df = continuous_df[(continuous_df['source'] == source) & (continuous_df['property'] == prop)]

        metrics, details_df = evaluate_regression(
            prop_df, predictions_cache, m['match']
        )

        if metrics is not None:
            metrics['source'] = source
            metrics['property'] = prop
            continuous_results.append(metrics)

            print(f"\n{source} / {prop}:")
            print(f"  Match: {metrics['endpoint_name']} ({metrics['match_quality']})")
            print(f"  N={metrics['n_samples']}, RMSE={metrics['rmse']:.3f}, R2={metrics['r2']:.3f}", end="")
            if metrics.get('pearson_r') is not None:
                print(f", r={metrics['pearson_r']:.3f}")
            else:
                print()

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

    # Overall summary
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    summary = {
        'timestamp': datetime.now().isoformat(),
        'evaluation_type': 'semantic_matching',
        'total_compounds_evaluated': len(predictions_cache),
        'total_compounds_failed': len(failed),
        'properties_matched': len(matched_properties),
        'properties_unmatched': len(unmatched_properties),
        'binary_classification': {
            'n_evaluations': len(binary_results),
            'mean_accuracy': binary_summary['accuracy'].mean() if binary_results else None,
            'mean_f1': binary_summary['f1'].mean() if binary_results else None,
            'mean_auc': auc_values.mean() if binary_results and len(auc_values) > 0 else None,
        },
        'continuous_regression': {
            'n_evaluations': len(continuous_results),
            'mean_rmse': continuous_summary['rmse'].mean() if continuous_results else None,
            'mean_r2': continuous_summary['r2'].mean() if continuous_results else None,
        }
    }

    with open(f"{RESULTS_DIR}/evaluation_summary.json", 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    print(f"\nResults saved to: {RESULTS_DIR}/")
    print(f"Completed: {datetime.now().isoformat()}")

    return summary


if __name__ == "__main__":
    run_semantic_evaluation()
