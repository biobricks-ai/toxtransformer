#!/usr/bin/env python3
"""
Run comprehensive adapter evaluation on external datasets.

Compares different feature selection methods:
- none: All 6,647 features
- topk: Top 500 statistically selected features
- semantic: LLM-selected relevant features
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

sys.path.insert(0, str(Path(__file__).parent.parent))

from adapter.feature_extractor import ToxTransformerFeatureExtractor
from adapter.trainer import AdapterTrainer
from adapter.evaluator import AdapterEvaluator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)


@dataclass
class EvalConfig:
    """Configuration for an evaluation endpoint."""
    name: str
    data_path: str
    label_col: str
    target_property: str  # For semantic selection
    target_description: Optional[str] = None
    is_binary: bool = True
    binarize_threshold: Optional[float] = None


ENDPOINTS = [
    EvalConfig(
        name="Ames Mutagenicity",
        data_path="cache/external_evaluation/ames_external.parquet",
        label_col="label",
        target_property="mutagenicity",
        target_description="Ames test for mutagenicity - bacterial reverse mutation assay detecting gene mutations",
    ),
    EvalConfig(
        name="Carcinogenicity",
        data_path="cache/external_evaluation/carcinogens_external.parquet",
        label_col="label",
        target_property="carcinogenicity",
        target_description="Carcinogenic potential - ability to cause cancer in animal or human studies",
    ),
    EvalConfig(
        name="LD50 (Acute Toxicity)",
        data_path="cache/external_evaluation/ld50_external.parquet",
        label_col="label",
        target_property="acute oral toxicity",
        target_description="Acute lethal dose (LD50) - dose causing 50% mortality in test animals",
        is_binary=False,
        binarize_threshold=2.5,  # log(LD50) < 2.5 = toxic
    ),
]


def load_data(config: EvalConfig) -> Tuple[List[str], np.ndarray]:
    """Load and prepare data for evaluation."""
    df = pd.read_parquet(config.data_path)

    inchis = df["inchi"].tolist()
    labels = df[config.label_col].values

    if not config.is_binary:
        # Binarize continuous labels
        labels = (labels < config.binarize_threshold).astype(int)
        logging.info(f"Binarized {config.name} at threshold {config.binarize_threshold}")

    return inchis, labels.astype(int)


def run_evaluation(
    config: EvalConfig,
    feature_selection: str,
    extractor: ToxTransformerFeatureExtractor,
    cache_features: Dict[str, np.ndarray],
    n_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Dict:
    """Run evaluation for a single configuration."""
    logging.info(f"\n{'='*60}")
    logging.info(f"Evaluating: {config.name} with {feature_selection} feature selection")
    logging.info(f"{'='*60}")

    # Load data
    inchis, labels = load_data(config)

    # Train/test split
    inchis_train, inchis_test, y_train, y_test = train_test_split(
        inchis, labels,
        test_size=test_size,
        stratify=labels,
        random_state=random_state,
    )

    logging.info(f"Train: {len(inchis_train)}, Test: {len(inchis_test)}")
    logging.info(f"Train positive rate: {y_train.mean():.2%}")

    # Get features (from cache or extract)
    cache_key = config.name
    if cache_key not in cache_features:
        logging.info("Extracting features...")
        all_features = extractor.batch_extract(inchis)
        cache_features[cache_key] = all_features
    else:
        all_features = cache_features[cache_key]

    # Split features
    train_indices = [inchis.index(i) for i in inchis_train]
    test_indices = [inchis.index(i) for i in inchis_test]
    X_train = all_features[train_indices]
    X_test = all_features[test_indices]

    # Create trainer
    trainer = AdapterTrainer(
        model_type="logistic",
        n_folds=n_folds,
        feature_selection=feature_selection,
        n_features=500,  # For topk
        random_state=random_state,
        target_property=config.target_property,
        target_description=config.target_description,
        semantic_max_categories=8,
        use_vertex_ai=True,  # Use Vertex AI for semantic selection
        vertex_project="insilica-internal",
    )

    # Train
    result = trainer.train_from_features(X_train, y_train)

    # Evaluate on test set
    evaluator = AdapterEvaluator()
    eval_result = evaluator.evaluate_from_features(result.model, X_test, y_test)

    # Get baseline
    baselines = evaluator.compute_baselines(y_test)

    return {
        "endpoint": config.name,
        "feature_selection": feature_selection,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "cv_auc": result.mean_auc,
        "cv_auc_std": result.std_auc,
        "test_auc": eval_result.auc,
        "test_accuracy": eval_result.accuracy,
        "test_f1": eval_result.f1,
        "baseline_auc": baselines.random_auc,
        "n_features_used": result.selected_features or result.n_features,
        "improvement": eval_result.auc - baselines.random_auc,
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Run adapter evaluation")
    parser.add_argument("--api-url", default=None, help="ToxTransformer API URL")
    parser.add_argument("--cache-path", default="cache/adapter_eval_features.sqlite")
    parser.add_argument("--output", default="cache/adapter_evaluation_results.csv")
    parser.add_argument("--skip-semantic", action="store_true", help="Skip semantic selection (requires API key)")
    args = parser.parse_args()

    # Initialize feature extractor
    extractor = ToxTransformerFeatureExtractor(
        api_url=args.api_url,
        cache_path=args.cache_path,
    )

    # Feature cache to avoid re-extraction
    feature_cache = {}

    # Feature selection methods to test
    if args.skip_semantic:
        methods = ["none", "topk"]
    else:
        methods = ["none", "topk", "semantic"]

    # Run all evaluations
    all_results = []

    for config in ENDPOINTS:
        for method in methods:
            try:
                result = run_evaluation(
                    config=config,
                    feature_selection=method,
                    extractor=extractor,
                    cache_features=feature_cache,
                )
                all_results.append(result)

                logging.info(
                    f"  {config.name} ({method}): "
                    f"CV AUC={result['cv_auc']:.3f}, "
                    f"Test AUC={result['test_auc']:.3f}, "
                    f"Features={result['n_features_used']}"
                )

            except Exception as e:
                logging.error(f"Error evaluating {config.name} with {method}: {e}")
                import traceback
                traceback.print_exc()

    # Create results dataframe
    results_df = pd.DataFrame(all_results)

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results_df.to_csv(output_path, index=False)
    logging.info(f"\nResults saved to {output_path}")

    # Print summary table
    print("\n" + "=" * 100)
    print("ADAPTER EVALUATION RESULTS")
    print("=" * 100)

    # Pivot for nice display
    pivot = results_df.pivot_table(
        index="endpoint",
        columns="feature_selection",
        values=["test_auc", "n_features_used"],
        aggfunc="first",
    )

    print("\nTest AUC by Endpoint and Feature Selection:")
    print("-" * 80)

    for endpoint in results_df["endpoint"].unique():
        print(f"\n{endpoint}:")
        endpoint_results = results_df[results_df["endpoint"] == endpoint]
        for _, row in endpoint_results.iterrows():
            print(
                f"  {row['feature_selection']:12s}: "
                f"AUC={row['test_auc']:.4f} "
                f"(CV: {row['cv_auc']:.4f}±{row['cv_auc_std']:.4f}) "
                f"[{row['n_features_used']} features]"
            )

    # Summary table
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"\n{'Endpoint':<25} {'Method':<12} {'Test AUC':>10} {'CV AUC':>10} {'Features':>10} {'vs Baseline':>12}")
    print("-" * 80)

    for _, row in results_df.iterrows():
        print(
            f"{row['endpoint']:<25} "
            f"{row['feature_selection']:<12} "
            f"{row['test_auc']:>10.4f} "
            f"{row['cv_auc']:>10.4f} "
            f"{row['n_features_used']:>10} "
            f"{row['improvement']:>+12.4f}"
        )

    return results_df


if __name__ == "__main__":
    main()
