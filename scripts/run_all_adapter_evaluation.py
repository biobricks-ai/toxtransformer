#!/usr/bin/env python3
"""
Run adapter evaluation on ALL viable binary classification endpoints.

Compares feature selection methods:
- none: All 6,647 features
- topk: Top 500 statistically selected features
- semantic: LLM-selected relevant features
"""

import json
import logging
import sys
import time
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
class EndpointConfig:
    """Configuration for an evaluation endpoint."""
    source: str
    property_path: str
    name: str
    n_samples: int
    n_positive: int
    n_negative: int
    target_property: str
    target_description: Optional[str] = None


def get_viable_endpoints(
    df: pd.DataFrame,
    min_samples: int = 30,
    min_per_class: int = 10,
) -> List[EndpointConfig]:
    """Get all viable endpoints from the benchmark data."""
    endpoints = []

    for (source, prop), group in df.groupby(['source', 'property']):
        n_total = len(group)
        n_pos = int(group['value'].sum())
        n_neg = n_total - n_pos

        if n_total >= min_samples and n_pos >= min_per_class and n_neg >= min_per_class:
            # Create readable name from property path
            name_parts = prop.split('/')
            if len(name_parts) >= 2:
                name = name_parts[-1]
            else:
                name = prop

            # Clean up name
            name = name.replace('_', ' ').title()

            # Create target property for semantic selection
            target_property = name.lower()

            # Add description based on source/property patterns
            description = None
            prop_lower = prop.lower()
            if 'ames' in prop_lower:
                description = "Ames test for bacterial mutagenicity"
            elif 'herg' in prop_lower:
                description = "hERG potassium channel inhibition (cardiac toxicity)"
            elif 'cyp' in prop_lower:
                description = "Cytochrome P450 enzyme inhibition (drug metabolism)"
            elif 'liver' in prop_lower or 'hepato' in prop_lower:
                description = "Liver toxicity / hepatotoxicity"
            elif 'cardio' in prop_lower:
                description = "Cardiovascular toxicity"
            elif 'carcino' in prop_lower:
                description = "Carcinogenicity - cancer causing potential"
            elif 'pgp' in prop_lower or 'p-gp' in prop_lower:
                description = "P-glycoprotein substrate (drug efflux transporter)"
            elif 'bbb' in prop_lower:
                description = "Blood-brain barrier penetration"
            elif 'skin' in prop_lower:
                description = "Skin sensitization / allergic reaction"
            elif 'zebrafish' in prop_lower:
                description = "Zebrafish developmental toxicity"
            elif 'dili' in prop_lower:
                description = "Drug-induced liver injury"
            elif 'renal' in prop_lower:
                description = "Kidney / renal toxicity"
            elif 'pulmonary' in prop_lower:
                description = "Lung / pulmonary toxicity"
            elif 'hematological' in prop_lower:
                description = "Blood / hematological toxicity"
            elif 'ototoxicity' in prop_lower:
                description = "Hearing / ear toxicity"
            elif 'infertility' in prop_lower:
                description = "Reproductive toxicity / infertility"
            elif 'dermatological' in prop_lower:
                description = "Skin / dermatological toxicity"
            elif 'hia' in prop_lower:
                description = "Human intestinal absorption"
            elif 'aquatic' in prop_lower or 'ecotox' in prop_lower:
                description = "Aquatic/environmental toxicity"

            endpoints.append(EndpointConfig(
                source=source,
                property_path=prop,
                name=f"{source}/{name}",
                n_samples=n_total,
                n_positive=n_pos,
                n_negative=n_neg,
                target_property=target_property,
                target_description=description,
            ))

    # Sort by sample size (largest first)
    endpoints.sort(key=lambda x: x.n_samples, reverse=True)
    return endpoints


def run_evaluation(
    config: EndpointConfig,
    df: pd.DataFrame,
    feature_selection: str,
    extractor: ToxTransformerFeatureExtractor,
    cache_features: Dict[str, np.ndarray],
    n_folds: int = 5,
    test_size: float = 0.2,
    random_state: int = 42,
    max_samples: int = 500,
) -> Optional[Dict]:
    """Run evaluation for a single configuration."""

    # Get data for this endpoint
    mask = (df['source'] == config.source) & (df['property'] == config.property_path)
    endpoint_df = df[mask].copy()

    if len(endpoint_df) < 30:
        return None

    # Sample if too large (balanced sampling)
    if len(endpoint_df) > max_samples:
        pos_df = endpoint_df[endpoint_df['value'] == 1]
        neg_df = endpoint_df[endpoint_df['value'] == 0]
        n_per_class = max_samples // 2
        n_pos = min(len(pos_df), n_per_class)
        n_neg = min(len(neg_df), n_per_class)
        # Adjust to maintain balance if one class is smaller
        if n_pos < n_per_class:
            n_neg = min(len(neg_df), max_samples - n_pos)
        elif n_neg < n_per_class:
            n_pos = min(len(pos_df), max_samples - n_neg)
        endpoint_df = pd.concat([
            pos_df.sample(n=n_pos, random_state=random_state),
            neg_df.sample(n=n_neg, random_state=random_state),
        ])
        logging.info(f"    Sampled to {len(endpoint_df)} ({n_pos}+ / {n_neg}-)")

    inchis = endpoint_df['inchi'].tolist()
    labels = endpoint_df['value'].values.astype(int)

    # Train/test split
    try:
        inchis_train, inchis_test, y_train, y_test = train_test_split(
            inchis, labels,
            test_size=test_size,
            stratify=labels,
            random_state=random_state,
        )
    except ValueError as e:
        logging.warning(f"Could not stratify {config.name}: {e}")
        return None

    # Get features (from cache or extract)
    cache_key = config.name
    if cache_key not in cache_features:
        all_features = extractor.batch_extract(inchis)
        cache_features[cache_key] = all_features
    else:
        all_features = cache_features[cache_key]

    # Check for failed extractions
    if all_features is None or len(all_features) == 0:
        return None

    # Split features
    try:
        train_indices = [inchis.index(i) for i in inchis_train]
        test_indices = [inchis.index(i) for i in inchis_test]
        X_train = all_features[train_indices]
        X_test = all_features[test_indices]
    except (ValueError, IndexError) as e:
        logging.warning(f"Index error for {config.name}: {e}")
        return None

    # Create trainer
    trainer = AdapterTrainer(
        model_type="logistic",
        n_folds=n_folds,
        feature_selection=feature_selection,
        n_features=500,
        random_state=random_state,
        target_property=config.target_property,
        target_description=config.target_description,
        semantic_max_categories=8,
        use_vertex_ai=True,
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
        "source": config.source,
        "property": config.property_path,
        "endpoint": config.name,
        "feature_selection": feature_selection,
        "n_train": len(y_train),
        "n_test": len(y_test),
        "n_positive": config.n_positive,
        "n_negative": config.n_negative,
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

    parser = argparse.ArgumentParser(description="Run adapter evaluation on all endpoints")
    parser.add_argument("--api-url", default=None, help="ToxTransformer API URL")
    parser.add_argument("--cache-path", default="cache/adapter_all_eval_features.sqlite")
    parser.add_argument("--output", default="cache/adapter_all_evaluation_results.csv")
    parser.add_argument("--skip-semantic", action="store_true", help="Skip semantic selection")
    parser.add_argument("--max-endpoints", type=int, default=None, help="Max endpoints to evaluate")
    parser.add_argument("--resume-from", type=int, default=0, help="Resume from endpoint index")
    parser.add_argument("--max-samples", type=int, default=500, help="Max samples per endpoint")
    args = parser.parse_args()

    # Load benchmark data
    data_path = "publication/benchmark/data/external_binary_benchmark.parquet"
    logging.info(f"Loading data from {data_path}")
    df = pd.read_parquet(data_path)
    logging.info(f"Loaded {len(df):,} records")

    # Get viable endpoints
    endpoints = get_viable_endpoints(df)
    logging.info(f"Found {len(endpoints)} viable endpoints")

    if args.max_endpoints:
        endpoints = endpoints[:args.max_endpoints]
        logging.info(f"Limited to {len(endpoints)} endpoints")

    if args.resume_from > 0:
        endpoints = endpoints[args.resume_from:]
        logging.info(f"Resuming from index {args.resume_from}, {len(endpoints)} remaining")

    # Initialize feature extractor
    extractor = ToxTransformerFeatureExtractor(
        api_url=args.api_url,
        cache_path=args.cache_path,
    )

    # Feature cache
    feature_cache = {}

    # Feature selection methods
    if args.skip_semantic:
        methods = ["none", "topk"]
    else:
        methods = ["none", "topk", "semantic"]

    # Load existing results if resuming
    output_path = Path(args.output)
    existing_results = []
    if output_path.exists() and args.resume_from > 0:
        existing_df = pd.read_csv(output_path)
        existing_results = existing_df.to_dict('records')
        logging.info(f"Loaded {len(existing_results)} existing results")

    # Run evaluations
    all_results = existing_results.copy()
    start_time = time.time()

    for i, config in enumerate(endpoints):
        endpoint_start = time.time()
        logging.info(f"\n{'='*60}")
        logging.info(f"[{i+1}/{len(endpoints)}] {config.name}")
        logging.info(f"  Samples: {config.n_samples} ({config.n_positive}+ / {config.n_negative}-)")
        logging.info(f"{'='*60}")

        for method in methods:
            try:
                logging.info(f"  Running {method} feature selection...")
                result = run_evaluation(
                    config=config,
                    df=df,
                    feature_selection=method,
                    extractor=extractor,
                    cache_features=feature_cache,
                    max_samples=args.max_samples,
                )

                if result:
                    all_results.append(result)
                    logging.info(
                        f"    {method}: CV AUC={result['cv_auc']:.3f}, "
                        f"Test AUC={result['test_auc']:.3f}, "
                        f"Features={result['n_features_used']}"
                    )
                else:
                    logging.warning(f"    {method}: No result")

            except Exception as e:
                logging.error(f"    {method}: Error - {e}")
                import traceback
                traceback.print_exc()

        # Save intermediate results
        if (i + 1) % 5 == 0 or i == len(endpoints) - 1:
            results_df = pd.DataFrame(all_results)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            results_df.to_csv(output_path, index=False)
            logging.info(f"  Saved {len(all_results)} results to {output_path}")

        endpoint_time = time.time() - endpoint_start
        logging.info(f"  Endpoint completed in {endpoint_time:.1f}s")

    # Final save
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_path, index=False)

    total_time = time.time() - start_time
    logging.info(f"\n{'='*100}")
    logging.info(f"EVALUATION COMPLETE")
    logging.info(f"{'='*100}")
    logging.info(f"Total endpoints: {len(endpoints)}")
    logging.info(f"Total results: {len(all_results)}")
    logging.info(f"Total time: {total_time/60:.1f} minutes")
    logging.info(f"Results saved to {output_path}")

    # Print summary
    if len(all_results) > 0:
        print("\n" + "=" * 100)
        print("SUMMARY BY FEATURE SELECTION METHOD")
        print("=" * 100)

        for method in methods:
            method_results = [r for r in all_results if r['feature_selection'] == method]
            if method_results:
                aucs = [r['test_auc'] for r in method_results]
                print(f"\n{method.upper()}:")
                print(f"  Endpoints: {len(method_results)}")
                print(f"  Mean AUC: {np.mean(aucs):.4f}")
                print(f"  Median AUC: {np.median(aucs):.4f}")
                print(f"  AUC >= 0.70: {sum(1 for a in aucs if a >= 0.70)}")
                print(f"  AUC >= 0.65: {sum(1 for a in aucs if a >= 0.65)}")

    return results_df


if __name__ == "__main__":
    main()
