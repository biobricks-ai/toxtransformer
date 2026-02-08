#!/usr/bin/env python3
"""
Validate Adapter Models

Trains and evaluates adapter models on UniTox endpoints to validate
the adapter approach. Compares adapter AUC vs direct ToxTransformer prediction AUC.
"""

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter.evaluator import AdapterEvaluator, format_comparison_table
from adapter.feature_extractor import ToxTransformerFeatureExtractor
from adapter.models import create_adapter
from adapter.trainer import AdapterTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)


@dataclass
class ValidationResult:
    """Result from validating adapter on an endpoint."""
    endpoint: str
    n_samples: int
    n_positive: int
    adapter_auc: float
    adapter_std: float
    baseline_auc: float
    direct_auc: Optional[float]
    improvement: float
    model_type: str


def load_unitox_data(
    property_col: str = "liver_toxicity_binary_rating_0_1",
) -> Tuple[List[str], List[int]]:
    """Load UniTox prepared data."""
    data_path = Path("cache/external_eval/unitox_prepared.csv")
    if not data_path.exists():
        raise FileNotFoundError(
            f"UniTox data not found at {data_path}. "
            "Run scripts/create_external_evaluation.py first."
        )

    df = pd.read_csv(data_path)

    if property_col not in df.columns:
        available = [
            c for c in df.columns if c not in ["inchi", "smiles", "compound_name"]
        ]
        raise ValueError(f"Property '{property_col}' not found. Available: {available}")

    # Filter to rows with this property
    mask = df[property_col].notna()
    df = df[mask].copy()

    inchis = df["inchi"].tolist()
    labels = df[property_col].astype(int).tolist()

    return inchis, labels


def get_direct_auc(
    inchis: List[str],
    labels: List[int],
    property_tokens: List[int],
    extractor: ToxTransformerFeatureExtractor,
) -> float:
    """
    Compute AUC using direct ToxTransformer predictions.

    Uses the average of specified property token predictions.
    """
    from sklearn.metrics import roc_auc_score

    # Extract features
    features = extractor.batch_extract(inchis, show_progress=False)

    # Get token indices
    tokens = extractor.get_property_tokens()
    if tokens is None:
        return 0.5

    token_to_idx = {t: i for i, t in enumerate(tokens)}

    # Get predictions for specified tokens
    indices = [token_to_idx.get(t) for t in property_tokens if t in token_to_idx]
    if not indices:
        return 0.5

    # Average predictions across tokens
    predictions = features[:, indices].mean(axis=1)

    try:
        return roc_auc_score(labels, predictions)
    except ValueError:
        return 0.5


def validate_endpoint(
    endpoint: str,
    inchis: List[str],
    labels: List[int],
    model_type: str = "logistic",
    direct_tokens: Optional[List[int]] = None,
    api_url: Optional[str] = None,
    cache_path: str = "cache/adapter_features.sqlite",
    n_folds: int = 5,
    feature_selection: str = "none",
    n_features: int = 1000,
    random_state: int = 42,
) -> ValidationResult:
    """
    Validate adapter on a single endpoint.

    Args:
        endpoint: Endpoint name.
        inchis: List of InChI strings.
        labels: List of binary labels.
        model_type: "logistic" or "xgboost".
        direct_tokens: Token IDs for direct AUC comparison.
        api_url: ToxTransformer API URL.
        cache_path: Path to feature cache.
        n_folds: Number of CV folds.
        feature_selection: Feature selection method.
        n_features: Number of features for topk selection.
        random_state: Random seed.

    Returns:
        ValidationResult with metrics.
    """
    logging.info(f"\n{'=' * 60}")
    logging.info(f"Validating endpoint: {endpoint}")
    logging.info(f"Samples: {len(inchis)}, Positive: {sum(labels)}")
    logging.info(f"{'=' * 60}")

    # Create trainer
    trainer = AdapterTrainer(
        model_type=model_type,
        n_folds=n_folds,
        feature_selection=feature_selection,
        n_features=n_features,
        api_url=api_url,
        cache_path=cache_path,
        random_state=random_state,
    )

    # Train with CV
    result = trainer.train(inchis, labels)

    # Compute baseline
    evaluator = AdapterEvaluator(api_url=api_url, cache_path=cache_path)
    baselines = evaluator.compute_baselines(np.array(labels))

    # Compute direct AUC if tokens provided
    direct_auc = None
    if direct_tokens:
        logging.info(f"Computing direct AUC for tokens: {direct_tokens}")
        direct_auc = get_direct_auc(
            inchis, labels, direct_tokens, trainer.extractor
        )
        logging.info(f"Direct AUC: {direct_auc:.4f}")

    # Compute improvement
    improvement = result.mean_auc - baselines.random_auc

    logging.info(f"\nResults for {endpoint}:")
    logging.info(f"  Adapter AUC:  {result.mean_auc:.4f} +/- {result.std_auc:.4f}")
    logging.info(f"  Baseline AUC: {baselines.random_auc:.4f}")
    if direct_auc:
        logging.info(f"  Direct AUC:   {direct_auc:.4f}")
    logging.info(f"  Improvement:  {improvement:+.4f}")

    return ValidationResult(
        endpoint=endpoint,
        n_samples=len(inchis),
        n_positive=sum(labels),
        adapter_auc=result.mean_auc,
        adapter_std=result.std_auc,
        baseline_auc=baselines.random_auc,
        direct_auc=direct_auc,
        improvement=improvement,
        model_type=model_type,
    )


def run_validation(
    endpoints: Optional[Dict[str, dict]] = None,
    model_types: List[str] = ["logistic"],
    feature_selections: List[str] = ["none"],
    api_url: Optional[str] = None,
    cache_path: str = "cache/adapter_features.sqlite",
    output_dir: str = "cache/adapter_validation",
) -> List[ValidationResult]:
    """
    Run full validation on multiple endpoints.

    Args:
        endpoints: Dict mapping endpoint name to config dict with keys:
            - property_col: Column name in UniTox data
            - direct_tokens: Optional list of ToxTransformer tokens for comparison
        model_types: List of model types to try.
        feature_selections: List of feature selection methods to try.
        api_url: ToxTransformer API URL.
        cache_path: Path to feature cache.
        output_dir: Directory to save results.

    Returns:
        List of ValidationResult objects.
    """
    if endpoints is None:
        # Default endpoints from UniTox
        endpoints = {
            "liver_toxicity": {
                "property_col": "liver_toxicity_binary_rating_0_1",
                "direct_tokens": None,  # No direct match
            },
            "cardio_toxicity": {
                "property_col": "cardio_toxicity_binary_rating_0_1",
                "direct_tokens": [1757, 2084, 2239],  # hERG related tokens
            },
            "kidney_toxicity": {
                "property_col": "kidney_toxicity_binary_rating_0_1",
                "direct_tokens": None,
            },
        }

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    all_results = []

    for model_type in model_types:
        for feature_selection in feature_selections:
            logging.info(f"\n\n{'#' * 60}")
            logging.info(f"Model: {model_type}, Feature Selection: {feature_selection}")
            logging.info(f"{'#' * 60}")

            for name, config in endpoints.items():
                try:
                    inchis, labels = load_unitox_data(config["property_col"])

                    result = validate_endpoint(
                        endpoint=name,
                        inchis=inchis,
                        labels=labels,
                        model_type=model_type,
                        direct_tokens=config.get("direct_tokens"),
                        api_url=api_url,
                        cache_path=cache_path,
                        feature_selection=feature_selection,
                    )

                    all_results.append(result)

                except Exception as e:
                    logging.error(f"Error validating {name}: {e}")
                    continue

    # Save results
    results_df = pd.DataFrame([vars(r) for r in all_results])
    results_df.to_csv(output_path / "validation_results.csv", index=False)

    # Print summary table
    print("\n\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    print(results_df.to_string(index=False))

    # Compute success criteria
    successful = [r for r in all_results if r.adapter_auc > 0.65]
    print(f"\n\nEndpoints with AUC > 0.65: {len(successful)}/{len(all_results)}")

    if all_results:
        mean_improvement = np.mean([r.improvement for r in all_results])
        print(f"Mean improvement over baseline: {mean_improvement:+.4f}")

    return all_results


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Validate adapter models")
    parser.add_argument(
        "--api-url",
        default=None,
        help="ToxTransformer API URL",
    )
    parser.add_argument(
        "--cache-path",
        default="cache/adapter_features.sqlite",
        help="Path to feature cache",
    )
    parser.add_argument(
        "--output-dir",
        default="cache/adapter_validation",
        help="Output directory for results",
    )
    parser.add_argument(
        "--model",
        choices=["logistic", "xgboost", "both"],
        default="logistic",
        help="Model type(s) to evaluate",
    )
    parser.add_argument(
        "--feature-selection",
        choices=["none", "variance", "topk", "all"],
        default="none",
        help="Feature selection method(s)",
    )
    parser.add_argument(
        "--endpoint",
        help="Single endpoint to validate (optional)",
    )

    args = parser.parse_args()

    # Determine model types
    if args.model == "both":
        model_types = ["logistic", "xgboost"]
    else:
        model_types = [args.model]

    # Determine feature selections
    if args.feature_selection == "all":
        feature_selections = ["none", "variance", "topk"]
    else:
        feature_selections = [args.feature_selection]

    # Run validation
    if args.endpoint:
        # Single endpoint
        endpoints = {
            args.endpoint: {
                "property_col": f"{args.endpoint}_binary_rating_0_1",
                "direct_tokens": None,
            }
        }
    else:
        endpoints = None  # Use defaults

    run_validation(
        endpoints=endpoints,
        model_types=model_types,
        feature_selections=feature_selections,
        api_url=args.api_url,
        cache_path=args.cache_path,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
