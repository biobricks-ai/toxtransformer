#!/usr/bin/env python3
"""
Run ToxTransformer predictions on external evaluation datasets.
Evaluates mutagenicity, carcinogenicity, and LD50 predictions.
"""
import pandas as pd
import requests
import json
import time
from urllib.parse import quote
import sys
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, mean_squared_error
import numpy as np

API_URL = "http://136.111.102.10:6515"
EVAL_DIR = "cache/external_evaluation"

# Property mappings from evaluation_config.json
PROPERTY_MAPPING = {
    'ames': [2508, 1799, 3830, 2863],  # mutagenicity properties
    'carcinogens': [3574, 3128, 3291, 2881, 3191],  # carcinogenicity properties
    'ld50': [1181, 1152],  # LD50 properties
}

def predict_all(inchi, timeout=120):
    """Get all predictions for a compound."""
    try:
        url = f"{API_URL}/predict_all?inchi={quote(inchi, safe='')}"
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def predict_single(inchi, property_token, timeout=30):
    """Get single property prediction."""
    try:
        url = f"{API_URL}/predict?inchi={quote(inchi, safe='')}&property_token={property_token}"
        response = requests.get(url, timeout=timeout)
        if response.status_code == 200:
            return response.json().get('value')
        return None
    except:
        return None

def evaluate_binary_classification(df, property_tokens, endpoint_name, sample_size=50):
    """Evaluate binary classification endpoint (Ames, Carcinogenicity)."""
    print(f"\n=== Evaluating {endpoint_name} ===")
    print(f"Property tokens: {property_tokens}")

    # Sample for faster evaluation
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled to {sample_size} compounds")

    results = []
    for i, row in df.iterrows():
        inchi = row['inchi']
        expected = int(row['label'])

        print(f"  {len(results)+1}/{len(df)}...", end=" ", flush=True)

        # Get predictions for all matching property tokens
        predictions = predict_all(inchi, timeout=180)
        if predictions is None:
            print("FAILED (API)")
            continue

        # Find predictions for our target properties
        pred_values = []
        for pred in predictions:
            if pred.get('property_token') in property_tokens:
                pred_values.append(pred.get('value', 0))

        if not pred_values:
            print("No matching properties")
            continue

        # Average predictions across matching properties
        avg_pred = np.mean(pred_values)
        pred_binary = 1 if avg_pred >= 0.5 else 0

        results.append({
            'inchi': inchi,
            'expected': expected,
            'predicted_prob': avg_pred,
            'predicted_binary': pred_binary,
            'num_props': len(pred_values)
        })

        match = "MATCH" if pred_binary == expected else "MISS"
        print(f"OK - exp={expected}, pred={avg_pred:.3f} -> {pred_binary} ({match})")

    if not results:
        print("  No successful predictions!")
        return None

    # Calculate metrics
    results_df = pd.DataFrame(results)
    y_true = results_df['expected'].values
    y_pred_prob = results_df['predicted_prob'].values
    y_pred_binary = results_df['predicted_binary'].values

    accuracy = accuracy_score(y_true, y_pred_binary)
    f1 = f1_score(y_true, y_pred_binary)

    try:
        auc = roc_auc_score(y_true, y_pred_prob)
    except:
        auc = None

    print(f"\n  Results for {endpoint_name}:")
    print(f"    Samples: {len(results)}")
    print(f"    Accuracy: {accuracy:.3f}")
    print(f"    F1 Score: {f1:.3f}")
    if auc:
        print(f"    AUC-ROC: {auc:.3f}")

    # Save detailed results
    results_df.to_csv(f"{EVAL_DIR}/{endpoint_name}_results.csv", index=False)

    return {
        'endpoint': endpoint_name,
        'n_samples': len(results),
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc
    }

def evaluate_regression(df, property_tokens, endpoint_name, sample_size=50):
    """Evaluate regression endpoint (LD50)."""
    print(f"\n=== Evaluating {endpoint_name} ===")
    print(f"Property tokens: {property_tokens}")
    print("Note: LD50 is continuous - using regression metrics")

    # Sample for faster evaluation
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
        print(f"Sampled to {sample_size} compounds")

    results = []
    for i, row in df.iterrows():
        inchi = row['inchi']
        expected = float(row['label'])

        print(f"  {len(results)+1}/{len(df)}...", end=" ", flush=True)

        # Get predictions
        predictions = predict_all(inchi, timeout=180)
        if predictions is None:
            print("FAILED (API)")
            continue

        # Find predictions for our target properties
        pred_values = []
        for pred in predictions:
            if pred.get('property_token') in property_tokens:
                pred_values.append(pred.get('value', 0))

        if not pred_values:
            print("No matching properties")
            continue

        # Average predictions
        avg_pred = np.mean(pred_values)

        results.append({
            'inchi': inchi,
            'expected': expected,
            'predicted': avg_pred,
            'num_props': len(pred_values)
        })

        print(f"OK - exp={expected:.2f}, pred={avg_pred:.3f}")

    if not results:
        print("  No successful predictions!")
        return None

    # Calculate metrics
    results_df = pd.DataFrame(results)
    y_true = results_df['expected'].values
    y_pred = results_df['predicted'].values

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    correlation = np.corrcoef(y_true, y_pred)[0, 1]

    print(f"\n  Results for {endpoint_name}:")
    print(f"    Samples: {len(results)}")
    print(f"    RMSE: {rmse:.3f}")
    print(f"    Correlation: {correlation:.3f}")

    # Save detailed results
    results_df.to_csv(f"{EVAL_DIR}/{endpoint_name}_results.csv", index=False)

    return {
        'endpoint': endpoint_name,
        'n_samples': len(results),
        'rmse': rmse,
        'correlation': correlation
    }

def main():
    print("ToxTransformer External Evaluation")
    print("=" * 60)

    # Check service health
    print("\nChecking service health...")
    try:
        resp = requests.get(f"{API_URL}/health", timeout=10)
        if resp.status_code != 200:
            print(f"Service unhealthy: HTTP {resp.status_code}")
            sys.exit(1)
        print(f"  Service healthy: {resp.json()}")
    except Exception as e:
        print(f"  Service unreachable: {e}")
        sys.exit(1)

    all_results = []

    # Evaluate Ames mutagenicity
    try:
        ames_df = pd.read_parquet(f"{EVAL_DIR}/ames_external.parquet")
        result = evaluate_binary_classification(
            ames_df, PROPERTY_MAPPING['ames'], 'ames', sample_size=30
        )
        if result:
            all_results.append(result)
    except Exception as e:
        print(f"Ames evaluation failed: {e}")

    # Evaluate Carcinogenicity
    try:
        carc_df = pd.read_parquet(f"{EVAL_DIR}/carcinogens_external.parquet")
        result = evaluate_binary_classification(
            carc_df, PROPERTY_MAPPING['carcinogens'], 'carcinogens', sample_size=30
        )
        if result:
            all_results.append(result)
    except Exception as e:
        print(f"Carcinogenicity evaluation failed: {e}")

    # Evaluate LD50
    try:
        ld50_df = pd.read_parquet(f"{EVAL_DIR}/ld50_external.parquet")
        result = evaluate_regression(
            ld50_df, PROPERTY_MAPPING['ld50'], 'ld50', sample_size=30
        )
        if result:
            all_results.append(result)
    except Exception as e:
        print(f"LD50 evaluation failed: {e}")

    # Summary
    print("\n" + "=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)

    for result in all_results:
        print(f"\n{result['endpoint'].upper()}:")
        print(f"  Samples: {result['n_samples']}")
        if 'accuracy' in result:
            print(f"  Accuracy: {result['accuracy']:.3f}")
            print(f"  F1: {result['f1']:.3f}")
            if result.get('auc'):
                print(f"  AUC: {result['auc']:.3f}")
        else:
            print(f"  RMSE: {result['rmse']:.3f}")
            print(f"  Correlation: {result['correlation']:.3f}")

    # Save summary
    with open(f"{EVAL_DIR}/evaluation_summary.json", 'w') as f:
        json.dump(all_results, f, indent=2)

    print(f"\nDetailed results saved to {EVAL_DIR}/")

if __name__ == "__main__":
    main()
