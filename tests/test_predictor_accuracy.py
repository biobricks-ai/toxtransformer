"""
Test predictor accuracy against known values from the database.
Samples chemicals with known property values and validates predictions.
"""
import sqlite3
import numpy as np
import torch
from sklearn.metrics import roc_auc_score, accuracy_score
from collections import defaultdict
import sys
sys.path.insert(0, '.')

from flask_cvae.predictor import Predictor

def get_test_samples(db_path: str, n_samples: int = 500, min_props_per_chemical: int = 5):
    """
    Get test samples: chemicals with multiple known property values.
    Returns dict: {inchi: {property_token: value, ...}}
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row

    # Get chemicals with many properties (for good context)
    # Use LIMIT with OFFSET for reproducible sampling
    query = """
    WITH chemical_counts AS (
        SELECT inchi, COUNT(DISTINCT property_token) as prop_count
        FROM activity
        GROUP BY inchi
        HAVING prop_count >= ?
        LIMIT 10000
    )
    SELECT a.inchi, a.property_token, a.value
    FROM activity a
    INNER JOIN chemical_counts cc ON a.inchi = cc.inchi
    LIMIT ?
    """

    cursor = conn.execute(query, (min_props_per_chemical, n_samples * 20))

    samples = defaultdict(dict)
    for row in cursor:
        inchi = row['inchi']
        prop_token = int(row['property_token'])
        value = int(row['value'])
        samples[inchi][prop_token] = value

    conn.close()

    # Filter to chemicals with enough properties
    samples = {k: v for k, v in samples.items() if len(v) >= min_props_per_chemical}

    # Take first n_samples
    return dict(list(samples.items())[:n_samples])


def evaluate_predictor(predictor: Predictor, samples: dict, max_predictions: int = 2000):
    """
    Evaluate predictor accuracy on samples.
    For each chemical, predict properties and compare to known values.
    """
    all_true = []
    all_pred = []
    results_by_prop = defaultdict(lambda: {'true': [], 'pred': []})

    total_predictions = 0

    for inchi, props in samples.items():
        if total_predictions >= max_predictions:
            break

        # For each property, predict it
        for prop_token, true_value in props.items():
            if total_predictions >= max_predictions:
                break

            try:
                prediction = predictor.predict_property(inchi, prop_token)
                if prediction is None:
                    continue

                pred_value = prediction.value

                all_true.append(true_value)
                all_pred.append(pred_value)
                results_by_prop[prop_token]['true'].append(true_value)
                results_by_prop[prop_token]['pred'].append(pred_value)
                total_predictions += 1

            except Exception as e:
                print(f"Error predicting {inchi[:30]}... prop {prop_token}: {e}")
                continue

    return all_true, all_pred, results_by_prop


def calculate_metrics(y_true, y_pred):
    """Calculate AUC and accuracy metrics."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Binary predictions
    y_pred_binary = (y_pred > 0.5).astype(int)

    metrics = {
        'n_samples': len(y_true),
        'n_positive': int(y_true.sum()),
        'n_negative': int((1 - y_true).sum()),
    }

    # Only calculate AUC if we have both classes
    if len(np.unique(y_true)) > 1:
        metrics['auc'] = roc_auc_score(y_true, y_pred)
    else:
        metrics['auc'] = None

    metrics['accuracy'] = accuracy_score(y_true, y_pred_binary)
    metrics['mean_pred_positive'] = float(y_pred[y_true == 1].mean()) if y_true.sum() > 0 else None
    metrics['mean_pred_negative'] = float(y_pred[y_true == 0].mean()) if (1-y_true).sum() > 0 else None

    return metrics


def run_test(n_chemicals=100, max_predictions=1000):
    """Run the prediction accuracy test."""
    print("=" * 60)
    print("PREDICTOR ACCURACY TEST")
    print("=" * 60)

    print("\n1. Loading predictor...")
    predictor = Predictor(use_cache=False)  # Disable cache for fair test
    print(f"   Model loaded. {len(predictor.all_property_tokens)} properties available.")

    print(f"\n2. Sampling {n_chemicals} chemicals from database...")
    samples = get_test_samples('brick/cvae.sqlite', n_samples=n_chemicals, min_props_per_chemical=10)
    print(f"   Found {len(samples)} chemicals with sufficient properties.")

    total_props = sum(len(v) for v in samples.values())
    print(f"   Total property-value pairs: {total_props}")

    print(f"\n3. Running predictions (max {max_predictions})...")
    y_true, y_pred, by_prop = evaluate_predictor(predictor, samples, max_predictions=max_predictions)

    print(f"\n4. Results:")
    print("-" * 40)

    metrics = calculate_metrics(y_true, y_pred)

    print(f"   Total predictions: {metrics['n_samples']}")
    print(f"   Positive samples:  {metrics['n_positive']}")
    print(f"   Negative samples:  {metrics['n_negative']}")
    print()

    if metrics['auc'] is not None:
        print(f"   AUC:               {metrics['auc']:.4f}")
    else:
        print(f"   AUC:               N/A (single class)")

    print(f"   Accuracy:          {metrics['accuracy']:.4f}")
    print()

    if metrics['mean_pred_positive'] is not None:
        print(f"   Mean pred (true=1): {metrics['mean_pred_positive']:.4f}")
    if metrics['mean_pred_negative'] is not None:
        print(f"   Mean pred (true=0): {metrics['mean_pred_negative']:.4f}")

    # Per-property analysis (top properties with enough samples)
    print("\n5. Per-property AUC (properties with 20+ samples):")
    print("-" * 40)

    prop_metrics = []
    for prop_token, data in by_prop.items():
        if len(data['true']) >= 20:
            pm = calculate_metrics(data['true'], data['pred'])
            if pm['auc'] is not None:
                prop_metrics.append((prop_token, pm['auc'], pm['n_samples']))

    prop_metrics.sort(key=lambda x: x[1], reverse=True)

    for prop_token, auc, n in prop_metrics[:10]:
        print(f"   Property {prop_token:5d}: AUC={auc:.4f} (n={n})")

    if prop_metrics:
        median_auc = np.median([x[1] for x in prop_metrics])
        print(f"\n   Median property AUC: {median_auc:.4f}")

    print("\n" + "=" * 60)

    # Return metrics for programmatic use
    return {
        'overall': metrics,
        'per_property': prop_metrics
    }


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-chemicals', type=int, default=50, help='Number of chemicals to test')
    parser.add_argument('--max-predictions', type=int, default=500, help='Max predictions to run')
    args = parser.parse_args()

    results = run_test(n_chemicals=args.n_chemicals, max_predictions=args.max_predictions)

    # Exit with error if AUC is too low
    if results['overall']['auc'] is not None and results['overall']['auc'] < 0.6:
        print("\n⚠️  WARNING: AUC is below 0.6 threshold!")
        sys.exit(1)
    elif results['overall']['auc'] is not None and results['overall']['auc'] > 0.75:
        print("\n✓ PASS: AUC is above 0.75")
        sys.exit(0)
    else:
        print("\n⚠️  AUC is between 0.6-0.75, may need investigation")
        sys.exit(0)
