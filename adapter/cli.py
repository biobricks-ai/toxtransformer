"""
Adapter CLI

Command-line interface for training and evaluating adapter models.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from adapter.evaluator import AdapterEvaluator
from adapter.feature_extractor import ToxTransformerFeatureExtractor
from adapter.models import BaseAdapter, create_adapter
from adapter.trainer import AdapterTrainer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)


def load_data(
    source: Optional[str] = None,
    data_path: Optional[str] = None,
    property_col: Optional[str] = None,
    inchi_col: str = "inchi",
    label_col: Optional[str] = None,
) -> Tuple[List[str], List[int]]:
    """
    Load training data from a source or file.

    Args:
        source: Named data source (e.g., "unitox").
        data_path: Path to CSV file.
        property_col: Property column name for multi-property files.
        inchi_col: Column name for InChI strings.
        label_col: Column name for labels.

    Returns:
        Tuple of (inchis, labels).
    """
    if source == "unitox":
        # Load UniTox data from cache
        unitox_path = Path("cache/external_eval/unitox_prepared.csv")
        if not unitox_path.exists():
            raise FileNotFoundError(
                f"UniTox data not found at {unitox_path}. "
                "Run scripts/create_external_evaluation.py first."
            )

        df = pd.read_csv(unitox_path)

        if property_col is None:
            property_col = "liver_toxicity_binary_rating_0_1"

        if property_col not in df.columns:
            available = [c for c in df.columns if c not in ["inchi", "smiles"]]
            raise ValueError(
                f"Property '{property_col}' not found. Available: {available}"
            )

        # Filter to rows with this property
        mask = df[property_col].notna()
        df = df[mask].copy()

        inchis = df[inchi_col].tolist()
        labels = df[property_col].astype(int).tolist()

    elif data_path:
        df = pd.read_csv(data_path)

        if inchi_col not in df.columns:
            raise ValueError(f"InChI column '{inchi_col}' not found in data")

        if label_col is None:
            # Try to find a label column
            for col in ["label", "target", "y", "class"]:
                if col in df.columns:
                    label_col = col
                    break
            if label_col is None:
                raise ValueError("No label column found. Specify with --label-col")

        inchis = df[inchi_col].tolist()
        labels = df[label_col].astype(int).tolist()

    else:
        raise ValueError("Must specify either --source or --data")

    return inchis, labels


def cmd_train(args):
    """Train an adapter model."""
    logging.info("Loading training data...")
    inchis, labels = load_data(
        source=args.source,
        data_path=args.data,
        property_col=args.property,
        inchi_col=args.inchi_col,
        label_col=args.label_col,
    )

    logging.info(f"Loaded {len(inchis)} samples, {sum(labels)} positive")

    # Train/test split
    if args.test_size > 0:
        from sklearn.model_selection import train_test_split

        inchis_train, inchis_test, labels_train, labels_test = train_test_split(
            inchis,
            labels,
            test_size=args.test_size,
            stratify=labels,
            random_state=args.seed,
        )
        logging.info(
            f"Split: {len(inchis_train)} train, {len(inchis_test)} test"
        )
    else:
        inchis_train, labels_train = inchis, labels
        inchis_test, labels_test = None, None

    # Infer target property name for semantic selection if not provided
    target_name = getattr(args, "target_name", None)
    if args.feature_selection == "semantic" and target_name is None:
        # Try to infer from property column name
        if args.property:
            target_name = args.property.replace("_", " ").replace("binary rating 0 1", "").strip()
            logging.info(f"Inferred target property name: {target_name}")

    # Create trainer
    trainer = AdapterTrainer(
        model_type=args.model,
        n_folds=args.folds,
        feature_selection=args.feature_selection,
        n_features=args.n_features,
        api_url=args.api_url,
        cache_path=args.cache_path,
        random_state=args.seed,
        target_property=target_name,
        target_description=getattr(args, "target_description", None),
        semantic_model=getattr(args, "semantic_model", "gemini-2.0-flash"),
        semantic_max_categories=getattr(args, "semantic_max_categories", 10),
    )

    # Train
    result = trainer.train(inchis_train, labels_train)

    # Show semantic selection info if used
    if args.feature_selection == "semantic" and result.model.semantic_selection_result:
        ssr = result.model.semantic_selection_result
        print(f"\nSemantic Feature Selection:")
        print(f"  Categories selected: {ssr.selected_categories}")
        print(f"  Features selected: {ssr.n_selected}/{ssr.n_total}")
        print(f"  Reasoning: {ssr.reasoning}")

    # Print summary
    print("\n" + result.summary())

    # Save model
    if args.output:
        trainer.save_result(result, args.output)

    # Evaluate on test set
    if inchis_test:
        logging.info("Evaluating on test set...")
        evaluator = AdapterEvaluator(
            api_url=args.api_url,
            cache_path=args.cache_path,
        )
        eval_result = evaluator.evaluate(
            result.model,
            inchis_test,
            labels_test,
        )
        print("\n" + eval_result.summary())

        # Compare to baseline
        baselines = evaluator.compute_baselines(np.array(labels_test))
        print(f"\nBaseline AUC (random): {baselines.random_auc:.4f}")
        print(f"Baseline Accuracy (majority): {baselines.majority_accuracy:.4f}")
        print(f"AUC improvement: {eval_result.auc - baselines.random_auc:+.4f}")


def cmd_evaluate(args):
    """Evaluate a trained adapter model."""
    # Load model
    logging.info(f"Loading model from {args.model}...")
    model = BaseAdapter.load(args.model)

    # Load test data
    logging.info("Loading test data...")
    inchis, labels = load_data(
        source=args.source,
        data_path=args.data,
        property_col=args.property,
        inchi_col=args.inchi_col,
        label_col=args.label_col,
    )

    logging.info(f"Loaded {len(inchis)} test samples, {sum(labels)} positive")

    # Evaluate
    evaluator = AdapterEvaluator(
        api_url=args.api_url,
        cache_path=args.cache_path,
    )

    eval_result = evaluator.evaluate(model, inchis, labels)

    print("\n" + eval_result.summary())

    # Compare to baseline
    baselines = evaluator.compute_baselines(np.array(labels))
    print(f"\nBaseline AUC (random): {baselines.random_auc:.4f}")
    print(f"Baseline Accuracy (majority): {baselines.majority_accuracy:.4f}")
    print(f"AUC improvement: {eval_result.auc - baselines.random_auc:+.4f}")

    # Save predictions if requested
    if args.output:
        output_path = Path(args.output)
        predictions = pd.DataFrame(
            {
                "inchi": inchis,
                "y_true": eval_result.y_true,
                "y_pred": eval_result.y_pred,
                "y_proba": eval_result.y_proba,
            }
        )
        predictions.to_csv(output_path, index=False)
        logging.info(f"Saved predictions to {output_path}")


def cmd_extract(args):
    """Extract features for a dataset."""
    logging.info("Loading data...")
    inchis, labels = load_data(
        source=args.source,
        data_path=args.data,
        property_col=args.property,
        inchi_col=args.inchi_col,
        label_col=args.label_col,
    )

    logging.info(f"Extracting features for {len(inchis)} molecules...")
    extractor = ToxTransformerFeatureExtractor(
        api_url=args.api_url,
        cache_path=args.cache_path,
    )

    features = extractor.batch_extract(inchis)
    logging.info(f"Feature matrix shape: {features.shape}")

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".npz":
        np.savez_compressed(
            output_path,
            features=features,
            labels=np.array(labels),
            inchis=np.array(inchis),
        )
    elif output_path.suffix == ".npy":
        np.save(output_path, features)
    else:
        # Save as CSV
        df = pd.DataFrame(features)
        df.columns = [f"prop_{i}" for i in range(features.shape[1])]
        df["inchi"] = inchis
        df["label"] = labels
        df.to_csv(output_path, index=False)

    logging.info(f"Saved features to {output_path}")


def cmd_cache_stats(args):
    """Show feature cache statistics."""
    extractor = ToxTransformerFeatureExtractor(
        cache_path=args.cache_path,
    )
    stats = extractor.get_cache_stats()
    print(json.dumps(stats, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description="ToxTransformer Adapter CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # Common arguments
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--api-url",
        default=None,
        help="ToxTransformer API URL",
    )
    common.add_argument(
        "--cache-path",
        default="cache/adapter_features.sqlite",
        help="Path to feature cache",
    )
    common.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    # Data arguments
    data_args = argparse.ArgumentParser(add_help=False)
    data_args.add_argument(
        "--source",
        choices=["unitox"],
        help="Named data source",
    )
    data_args.add_argument(
        "--data",
        help="Path to CSV data file",
    )
    data_args.add_argument(
        "--property",
        help="Property column name",
    )
    data_args.add_argument(
        "--inchi-col",
        default="inchi",
        help="InChI column name",
    )
    data_args.add_argument(
        "--label-col",
        help="Label column name",
    )

    # Train command
    train_parser = subparsers.add_parser(
        "train",
        parents=[common, data_args],
        help="Train an adapter model",
    )
    train_parser.add_argument(
        "--model",
        choices=["logistic", "xgboost"],
        default="logistic",
        help="Model type",
    )
    train_parser.add_argument(
        "--folds",
        type=int,
        default=5,
        help="Number of CV folds",
    )
    train_parser.add_argument(
        "--feature-selection",
        choices=["none", "variance", "topk", "semantic"],
        default="none",
        help="Feature selection method. 'semantic' uses LLM to select relevant features.",
    )
    train_parser.add_argument(
        "--target-name",
        help="Target property name for semantic selection (e.g., 'liver toxicity')",
    )
    train_parser.add_argument(
        "--target-description",
        help="Optional description of target property for semantic selection",
    )
    train_parser.add_argument(
        "--semantic-model",
        default="gemini-2.0-flash",
        help="LLM model for semantic selection (default: gemini-2.0-flash)",
    )
    train_parser.add_argument(
        "--semantic-max-categories",
        type=int,
        default=10,
        help="Max categories for semantic selection (default: 10)",
    )
    train_parser.add_argument(
        "--n-features",
        type=int,
        default=1000,
        help="Number of features for topk selection",
    )
    train_parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Test set fraction (0 to skip)",
    )
    train_parser.add_argument(
        "--output",
        "-o",
        help="Output model path",
    )
    train_parser.set_defaults(func=cmd_train)

    # Evaluate command
    eval_parser = subparsers.add_parser(
        "evaluate",
        parents=[common, data_args],
        help="Evaluate a trained model",
    )
    eval_parser.add_argument(
        "--model",
        "-m",
        required=True,
        help="Path to trained model",
    )
    eval_parser.add_argument(
        "--output",
        "-o",
        help="Output predictions path",
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    # Extract command
    extract_parser = subparsers.add_parser(
        "extract",
        parents=[common, data_args],
        help="Extract features for a dataset",
    )
    extract_parser.add_argument(
        "--output",
        "-o",
        required=True,
        help="Output features path (.npz, .npy, or .csv)",
    )
    extract_parser.set_defaults(func=cmd_extract)

    # Cache stats command
    cache_parser = subparsers.add_parser(
        "cache-stats",
        parents=[common],
        help="Show cache statistics",
    )
    cache_parser.set_defaults(func=cmd_cache_stats)

    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
