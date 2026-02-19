#!/usr/bin/env python3
"""
Evaluate property linkages via simple-mean and distillation approaches.

Compares Claude-linked tokens vs old curated mappings from run_focused_eval.py.

Usage:
    python scripts/evaluate_linkages.py
    python scripts/evaluate_linkages.py --skip-distillation
    python scripts/evaluate_linkages.py --endpoints "admet-huggingface||admet-huggingface/admet/AMES"
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Ensure project root is on sys.path for adapter imports
_PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolFromInchi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

DEFAULT_LINKAGE_PATH = "cache/property_linkages.json"
DEFAULT_BENCHMARK_PATH = "publication/benchmark/data/external_binary_benchmark.parquet"
DEFAULT_CACHE_PATHS = [
    "cache/adapter_all_eval_features.sqlite",
    "cache/adapter_eval_features.sqlite",
]
DEFAULT_OUTPUT_PATH = "publication/benchmark/results/linkage_evaluation.csv"

MAX_SAMPLES_PER_CLASS = 250

# Old curated mappings from scripts/run_focused_eval.py for comparison
# Keys match the "source||property" format used in linkage JSON
CURATED_MAPPINGS = {
    "admet-huggingface||admet-huggingface/admet/AMES": [1799, 2508, 2863, 3830],
    "ames-benchmark||ames-benchmark/ames_cv_splits/ames_class": [1799, 2508, 2863, 3830],
    "chempile-tox||chempile-tox/tdc_ames/label": [1799, 2508, 2863, 3830],
    "admet-huggingface||admet-huggingface/admet/hERG": [213, 229, 1252, 1438, 1866, 2105, 2547],
    "cyp450||cyp450/cyp450_cyp1a2_combined/label": [1270, 1610, 5808, 5815],
    "cyp450||cyp450/cyp450_cyp2c9_combined/label": [701, 1582, 1854, 2219, 2800],
    "cyp450||cyp450/cyp450_cyp2c19_combined/label": [702, 1576, 2290],
    "cyp450||cyp450/cyp450_cyp2d6_combined/label": [1585, 2295],
    "cyp450||cyp450/cyp450_cyp3a4_combined/label": [1148, 1521, 1864, 2138],
    "carcinogens-lagunin||carcinogens-lagunin/carcinogens_lagunin/label": [2881, 3128, 3191, 3291, 3574],
    "admet-huggingface||admet-huggingface/admet/Carcinogens_Lagunin": [2881, 3128, 3191, 3291, 3574],
    "comptox-zebrafish||comptox-zebrafish/zebrafish_toxicity/mortality": [3418, 3455, 3487, 3488, 3489, 3490, 3492, 4075],
    "comptox-zebrafish||comptox-zebrafish/zebrafish_toxicity/activity_score": [3418, 3455, 3487, 3488, 3489, 3490, 3492, 4075],
    "unitox||unitox/unitox_small_molecules/liver_toxicity_binary_rating_0_1": [4438, 1294, 1298, 1300, 1386, 1404],
    "unitox||unitox/unitox_small_molecules/cardio_toxicity_binary_rating_0_1": [213, 229, 1252, 2105],
}


def load_linkages(path: str) -> Dict:
    """Load Claude-generated property linkages."""
    with open(path) as f:
        return json.load(f)


def balanced_sample(
    df: pd.DataFrame,
    max_per_class: int = MAX_SAMPLES_PER_CLASS,
    random_state: int = 42,
) -> pd.DataFrame:
    """Balanced sampling: min(max_per_class, n_minority) per class."""
    pos_df = df[df["value"] == 1]
    neg_df = df[df["value"] == 0]
    n_sample = min(max_per_class, len(pos_df), len(neg_df))

    if n_sample < 5:
        return pd.DataFrame()

    sampled = pd.concat([
        pos_df.sample(n=n_sample, random_state=random_state),
        neg_df.sample(n=n_sample, random_state=random_state),
    ])
    return sampled


def get_feature_extractor(cache_path: str):
    """Create a feature extractor with caching."""
    from adapter.feature_extractor import ToxTransformerFeatureExtractor
    return ToxTransformerFeatureExtractor(cache_path=cache_path, timeout=60)


def extract_features_for_inchis(
    extractor,
    inchis: List[str],
) -> Tuple[np.ndarray, List[int], List[bool]]:
    """
    Extract features for a list of InChIs using the feature extractor.

    Returns:
        feature_matrix: (n_molecules, 6647) array
        property_tokens: ordered list of property token IDs
        success_mask: boolean list indicating which InChIs succeeded
    """
    results = []
    success_mask = []

    for inchi in inchis:
        result = extractor.extract(inchi)
        results.append(result)
        success_mask.append(result.success)

    if not any(success_mask):
        return np.array([]), [], success_mask

    # Get property tokens from first successful result
    property_tokens = []
    for r in results:
        if r.success and r.property_tokens:
            property_tokens = r.property_tokens
            break

    # Stack features
    feature_matrix = np.vstack([r.features for r in results])
    return feature_matrix, property_tokens, success_mask


def compute_simple_mean_auc(
    feature_matrix: np.ndarray,
    property_tokens: List[int],
    y_true: np.ndarray,
    token_ids: List[int],
    success_mask: List[bool],
) -> Optional[float]:
    """Compute AUC using simple mean of selected token predictions."""
    if not token_ids:
        return None

    # Map token IDs to feature indices
    token_to_idx = {t: i for i, t in enumerate(property_tokens)}
    indices = [token_to_idx[t] for t in token_ids if t in token_to_idx]

    if not indices:
        return None

    # Filter to successful extractions
    mask = np.array(success_mask)
    X_valid = feature_matrix[mask]
    y_valid = y_true[mask]

    if len(y_valid) < 10 or len(set(y_valid)) < 2:
        return None

    # Simple mean of selected token values
    y_pred = X_valid[:, indices].mean(axis=1)

    try:
        return roc_auc_score(y_valid, y_pred)
    except ValueError:
        return None


def compute_distillation_auc(
    feature_matrix: np.ndarray,
    y_true: np.ndarray,
    success_mask: List[bool],
    n_folds: int = 5,
    random_state: int = 42,
) -> Optional[float]:
    """Compute AUC using logistic regression distillation on all 6,647 features."""
    from adapter.trainer import AdapterTrainer

    # Filter to successful extractions
    mask = np.array(success_mask)
    X_valid = feature_matrix[mask]
    y_valid = y_true[mask]

    if len(y_valid) < 20 or len(set(y_valid)) < 2:
        return None

    # Check minimum samples per class for CV
    n_pos = int(y_valid.sum())
    n_neg = len(y_valid) - n_pos
    if min(n_pos, n_neg) < n_folds:
        return None

    try:
        trainer = AdapterTrainer(
            model_type="logistic",
            n_folds=n_folds,
            feature_selection="none",
            random_state=random_state,
        )
        result = trainer.train_from_features(X_valid, y_valid)
        return result.mean_auc
    except Exception as e:
        log.warning(f"Distillation failed: {e}")
        return None


def compute_adapter_tree_auc(
    feature_matrix: np.ndarray,
    property_tokens: List[int],
    y_true: np.ndarray,
    token_ids: List[int],
    success_mask: List[bool],
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[Optional[float], Optional[str]]:
    """Compute AUC using a decision tree on Claude-linked tokens only.

    Uses the agentic prefilter (Claude-selected tokens) to reduce features
    from 6,647 to 1-7, then trains a shallow decision tree that can learn
    inversions, thresholds, and simple logic. Returns (auc, tree_rules).
    """
    from sklearn.tree import DecisionTreeClassifier, export_text

    if not token_ids:
        return None, None

    # Map token IDs to feature indices
    token_to_idx = {t: i for i, t in enumerate(property_tokens)}
    indices = [token_to_idx[t] for t in token_ids if t in token_to_idx]

    if not indices:
        return None, None

    # Filter to successful extractions
    mask = np.array(success_mask)
    X_valid = feature_matrix[mask][:, indices]
    y_valid = y_true[mask]

    if len(y_valid) < 10 or len(set(y_valid)) < 2:
        return None, None

    n_pos = int(y_valid.sum())
    n_neg = len(y_valid) - n_pos
    if min(n_pos, n_neg) < n_folds:
        return None, None

    # Limit tree depth based on number of features
    max_depth = min(3, len(indices))

    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_aucs = []
        for train_idx, test_idx in skf.split(X_valid, y_valid):
            X_train, X_test = X_valid[train_idx], X_valid[test_idx]
            y_train, y_test = y_valid[train_idx], y_valid[test_idx]

            if len(set(y_test)) < 2:
                continue

            clf = DecisionTreeClassifier(
                max_depth=max_depth,
                min_samples_leaf=max(3, len(y_train) // 10),
                random_state=random_state,
            )
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            fold_aucs.append(roc_auc_score(y_test, y_prob))

        if len(fold_aucs) == 0:
            return None, None

        auc = float(np.mean(fold_aucs))

        # Train final tree on all data for interpretability
        final_tree = DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_leaf=max(3, len(y_valid) // 10),
            random_state=random_state,
        )
        final_tree.fit(X_valid, y_valid)
        feature_names = [f"token_{t}" for t in token_ids if t in token_to_idx]
        rules = export_text(final_tree, feature_names=feature_names, decimals=3)

        return auc, rules
    except Exception as e:
        log.warning(f"Adapter tree failed: {e}")
        return None, None


def compute_adapter_logreg_auc(
    feature_matrix: np.ndarray,
    property_tokens: List[int],
    y_true: np.ndarray,
    token_ids: List[int],
    success_mask: List[bool],
    n_folds: int = 5,
    random_state: int = 42,
) -> Tuple[Optional[float], Optional[str]]:
    """Compute AUC using logistic regression on Claude-linked tokens only.

    L2-regularized logistic regression on 1-7 matched token features.
    Preserves calibrated probabilities, can learn weights and inversions
    via negative coefficients, and regularization prevents overfitting.
    Returns (auc, coefficients_description).
    """
    if not token_ids:
        return None, None

    # Map token IDs to feature indices
    token_to_idx = {t: i for i, t in enumerate(property_tokens)}
    indices = [token_to_idx[t] for t in token_ids if t in token_to_idx]

    if not indices:
        return None, None

    # Filter to successful extractions
    mask = np.array(success_mask)
    X_valid = feature_matrix[mask][:, indices]
    y_valid = y_true[mask]

    if len(y_valid) < 10 or len(set(y_valid)) < 2:
        return None, None

    n_pos = int(y_valid.sum())
    n_neg = len(y_valid) - n_pos
    if min(n_pos, n_neg) < n_folds:
        return None, None

    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_aucs = []
        for train_idx, test_idx in skf.split(X_valid, y_valid):
            X_train, X_test = X_valid[train_idx], X_valid[test_idx]
            y_train, y_test = y_valid[train_idx], y_valid[test_idx]

            if len(set(y_test)) < 2:
                continue

            clf = LogisticRegression(
                max_iter=1000, C=1.0, random_state=random_state,
                solver="lbfgs", penalty="l2",
            )
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            fold_aucs.append(roc_auc_score(y_test, y_prob))

        if len(fold_aucs) == 0:
            return None, None

        auc = float(np.mean(fold_aucs))

        # Train final model on all data for interpretability
        final_clf = LogisticRegression(
            max_iter=1000, C=1.0, random_state=random_state,
            solver="lbfgs", penalty="l2",
        )
        final_clf.fit(X_valid, y_valid)

        feature_names = [f"token_{t}" for t in token_ids if t in token_to_idx]
        coefs = final_clf.coef_[0]
        intercept = final_clf.intercept_[0]
        parts = [f"intercept={intercept:.3f}"]
        for name, coef in zip(feature_names, coefs):
            parts.append(f"{name}: {coef:+.3f}")
        desc = " | ".join(parts)

        return auc, desc
    except Exception as e:
        log.warning(f"Adapter logreg failed: {e}")
        return None, None


def compute_fingerprint_baseline_auc(
    inchis: List[str],
    y_true: np.ndarray,
    success_mask: List[bool],
    n_folds: int = 5,
    random_state: int = 42,
) -> Optional[float]:
    """Compute AUC using Morgan fingerprints + logistic regression baseline.

    Converts InChI -> mol -> Morgan FP (2048-bit, radius 2), then trains
    LogisticRegression with 5-fold StratifiedKFold CV and returns mean AUC.
    Molecules that fail RDKit parsing are excluded.
    """
    # Filter to molecules that succeeded in feature extraction
    mask = np.array(success_mask)
    inchis_valid = [inch for inch, m in zip(inchis, mask) if m]
    y_valid = y_true[mask]

    if len(y_valid) < 20 or len(set(y_valid)) < 2:
        return None

    # Convert InChI to Morgan fingerprints, tracking which ones succeed
    fps = []
    fp_mask = []
    for inchi in inchis_valid:
        mol = MolFromInchi(inchi, sanitize=True, removeHs=True)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fps.append(np.array(fp, dtype=np.float32))
            fp_mask.append(True)
        else:
            fp_mask.append(False)

    fp_mask = np.array(fp_mask)
    if fp_mask.sum() < 20:
        return None

    X_fp = np.vstack(fps)
    y_fp = y_valid[fp_mask]

    if len(set(y_fp)) < 2:
        return None

    # Check minimum samples per class for CV
    n_pos = int(y_fp.sum())
    n_neg = len(y_fp) - n_pos
    if min(n_pos, n_neg) < n_folds:
        return None

    try:
        skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
        fold_aucs = []
        for train_idx, test_idx in skf.split(X_fp, y_fp):
            X_train, X_test = X_fp[train_idx], X_fp[test_idx]
            y_train, y_test = y_fp[train_idx], y_fp[test_idx]

            if len(set(y_test)) < 2:
                continue

            clf = LogisticRegression(
                max_iter=1000, C=1.0, random_state=random_state, solver="lbfgs",
            )
            clf.fit(X_train, y_train)
            y_prob = clf.predict_proba(X_test)[:, 1]
            fold_aucs.append(roc_auc_score(y_test, y_prob))

        if len(fold_aucs) == 0:
            return None

        return float(np.mean(fold_aucs))
    except Exception as e:
        log.warning(f"Fingerprint baseline failed: {e}")
        return None


def evaluate_endpoint(
    endpoint_key: str,
    linkage: Dict,
    benchmark_df: pd.DataFrame,
    extractor,
    property_tokens: Optional[List[int]],
    feature_cache: Dict[str, np.ndarray],
    cache_only: bool = False,
) -> Optional[Dict]:
    """Evaluate a single endpoint."""
    source = linkage["source"]
    prop = linkage["property"]
    tokens = linkage.get("tokens", [])
    confidence = linkage.get("confidence", "unknown")

    # Find matching data
    mask = (benchmark_df["source"] == source) & (benchmark_df["property"] == prop)
    prop_df = benchmark_df[mask]

    if len(prop_df) < 20:
        log.info(f"  Skipping {endpoint_key}: only {len(prop_df)} samples")
        return None

    # In cache-only mode, restrict to molecules with cached features
    if cache_only:
        prop_df = prop_df[prop_df["inchi"].isin(feature_cache)]
        if len(prop_df) < 20:
            log.info(f"  Skipping {endpoint_key}: only {len(prop_df)} cached samples")
            return None

    # Balanced sampling
    sampled = balanced_sample(prop_df)
    if len(sampled) < 10:
        log.info(f"  Skipping {endpoint_key}: insufficient balanced samples")
        return None

    inchis = sampled["inchi"].tolist()
    y_true = sampled["value"].values.astype(int)
    n_pos = int(y_true.sum())
    n_neg = len(y_true) - n_pos

    log.info(f"  Evaluating {endpoint_key}: n={len(sampled)}, pos={n_pos}, neg={n_neg}")

    # Extract features (use cache when possible)
    features_list = []
    success_mask = []
    need_extract = []

    for inchi in inchis:
        if inchi in feature_cache:
            features_list.append(feature_cache[inchi])
            success_mask.append(True)
        else:
            need_extract.append(len(features_list))
            features_list.append(None)
            success_mask.append(False)

    # Extract missing features in parallel (skip in cache-only mode)
    if need_extract and not cache_only:
        from concurrent.futures import ThreadPoolExecutor, as_completed

        missing_inchis = [inchis[i] for i in need_extract]
        log.info(f"  Extracting features for {len(missing_inchis)} new molecules (parallel)...")

        def _extract_one(inchi):
            return inchi, extractor.extract(inchi)

        n_workers = min(8, len(missing_inchis))
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            futures = {pool.submit(_extract_one, inchi): idx
                       for idx, inchi in zip(need_extract, missing_inchis)}
            for future in as_completed(futures):
                idx = futures[future]
                inchi, result = future.result()
                if result.success:
                    features_list[idx] = result.features
                    success_mask[idx] = True
                    feature_cache[inchi] = result.features

                    # Set property_tokens from first successful extraction
                    if property_tokens is None or len(property_tokens) == 0:
                        property_tokens = result.property_tokens
                else:
                    features_list[idx] = np.zeros(extractor.NUM_PROPERTIES, dtype=np.float32)
                    success_mask[idx] = False
    elif need_extract and cache_only:
        # Should not happen since we filtered to cached inchis, but handle gracefully
        for idx in need_extract:
            features_list[idx] = np.zeros(extractor.NUM_PROPERTIES, dtype=np.float32)

    # Build feature matrix
    feature_matrix = np.vstack([f if f is not None else np.zeros(extractor.NUM_PROPERTIES, dtype=np.float32) for f in features_list])

    if not property_tokens:
        log.warning(f"  No property tokens available for {endpoint_key}")
        return None

    # Simple mean AUC with Claude-linked tokens
    simple_mean_auc = compute_simple_mean_auc(
        feature_matrix, property_tokens, y_true, tokens, success_mask,
    )

    # Adapter tree AUC (decision tree on Claude-linked tokens only)
    adapter_tree_auc, tree_rules = compute_adapter_tree_auc(
        feature_matrix, property_tokens, y_true, tokens, success_mask,
    )

    # Adapter logreg AUC (logistic regression on Claude-linked tokens only)
    adapter_logreg_auc, logreg_desc = compute_adapter_logreg_auc(
        feature_matrix, property_tokens, y_true, tokens, success_mask,
    )

    # Fingerprint baseline AUC
    fingerprint_auc = compute_fingerprint_baseline_auc(inchis, y_true, success_mask)

    # Delta: adapter logreg vs fingerprint baseline
    delta_logreg_vs_fp = None
    if adapter_logreg_auc is not None and fingerprint_auc is not None:
        delta_logreg_vs_fp = adapter_logreg_auc - fingerprint_auc

    return {
        "endpoint": endpoint_key,
        "source": source,
        "property": prop,
        "confidence": confidence,
        "n_tokens": len(tokens),
        "n_samples": len(sampled),
        "n_pos": n_pos,
        "n_evaluated": int(sum(success_mask)),
        "simple_mean_auc": round(simple_mean_auc, 4) if simple_mean_auc is not None else None,
        "adapter_tree_auc": round(adapter_tree_auc, 4) if adapter_tree_auc is not None else None,
        "adapter_logreg_auc": round(adapter_logreg_auc, 4) if adapter_logreg_auc is not None else None,
        "fingerprint_auc": round(fingerprint_auc, 4) if fingerprint_auc is not None else None,
        "delta_logreg_vs_fp": round(delta_logreg_vs_fp, 4) if delta_logreg_vs_fp is not None else None,
        "tree_rules": tree_rules,
        "logreg_desc": logreg_desc,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate property linkages")
    parser.add_argument("--linkages", default=DEFAULT_LINKAGE_PATH, help="Path to linkages JSON")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK_PATH, help="Path to benchmark parquet")
    parser.add_argument("--cache", type=str, action="append", help="Feature cache SQLite path(s). Can be repeated. Defaults to both adapter caches.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Output CSV path")
    parser.add_argument("--endpoints", type=str, help="Comma-separated endpoint keys (default: all)")
    parser.add_argument("--skip-distillation", action="store_true", help="(deprecated, ignored)")
    parser.add_argument("--cache-only", action="store_true", help="Only use cached features, skip API calls. Endpoints without enough cached molecules are skipped.")
    args = parser.parse_args()

    cache_paths = args.cache if args.cache else DEFAULT_CACHE_PATHS

    # Load linkages
    if not Path(args.linkages).exists():
        log.error(f"Linkages file not found: {args.linkages}")
        log.error("Run scripts/link_properties.py first.")
        sys.exit(1)

    linkages = load_linkages(args.linkages)
    log.info(f"Loaded {len(linkages)} linkages from {args.linkages}")

    # Filter endpoints
    if args.endpoints:
        filter_keys = [k.strip() for k in args.endpoints.split(",")]
        linkages = {k: v for k, v in linkages.items() if k in filter_keys}
        log.info(f"Filtered to {len(linkages)} endpoints")

    # Load benchmark data
    benchmark_df = pd.read_parquet(args.benchmark)
    log.info(f"Loaded {len(benchmark_df):,} benchmark records")

    # Initialize feature extractor (use first available cache for new extractions)
    primary_cache = next((p for p in cache_paths if Path(p).exists()), cache_paths[0])
    extractor = get_feature_extractor(primary_cache)

    # Pre-load cached features from ALL cache files into memory
    feature_cache: Dict[str, np.ndarray] = {}
    property_tokens: Optional[List[int]] = None

    import sqlite3
    for cache_path in cache_paths:
        if Path(cache_path).exists():
            conn = sqlite3.connect(cache_path)
            cursor = conn.execute("SELECT inchi, features, property_tokens FROM feature_cache")
            n_loaded = 0
            for row in cursor:
                inchi = row[0]
                if inchi not in feature_cache:
                    features = np.frombuffer(row[1], dtype=np.float32)
                    feature_cache[inchi] = features
                    n_loaded += 1
                    if property_tokens is None:
                        property_tokens = json.loads(row[2])
            conn.close()
            log.info(f"Loaded {n_loaded} new feature vectors from {cache_path}")

    log.info(f"Total cached features in memory: {len(feature_cache)} molecules")
    if args.cache_only:
        log.info("Cache-only mode: will only use cached features, no API calls")

    # Evaluate each endpoint
    results = []
    skipped_none = 0

    # Sort: process endpoints with tokens first
    sorted_keys = sorted(
        linkages.keys(),
        key=lambda k: (linkages[k].get("confidence") == "none", k),
    )

    for i, key in enumerate(sorted_keys):
        linkage = linkages[key]
        confidence = linkage.get("confidence", "unknown")
        n_tokens = len(linkage.get("tokens", []))

        log.info(f"\n[{i+1}/{len(sorted_keys)}] {key} (confidence={confidence}, tokens={n_tokens})")

        if confidence == "excluded":
            skipped_none += 1
            log.info(f"  Skipping (excluded — no valid tokens in catalog)")
            continue

        if confidence == "none" and n_tokens == 0:
            skipped_none += 1
            log.info(f"  Skipping (confidence=none, no tokens)")
            continue

        result = evaluate_endpoint(
            key, linkage, benchmark_df, extractor,
            property_tokens, feature_cache,
            cache_only=args.cache_only,
        )

        if result:
            results.append(result)
            sm_str = f"{result['simple_mean_auc']:.4f}" if result["simple_mean_auc"] is not None else "N/A"
            tree_str = f"{result['adapter_tree_auc']:.4f}" if result["adapter_tree_auc"] is not None else "N/A"
            lr_str = f"{result['adapter_logreg_auc']:.4f}" if result["adapter_logreg_auc"] is not None else "N/A"
            fp_str = f"{result['fingerprint_auc']:.4f}" if result["fingerprint_auc"] is not None else "N/A"
            log.info(f"  AUC: simple_mean={sm_str}, tree={tree_str}, logreg={lr_str}, fingerprint={fp_str}")

    # Save results
    if results:
        results_df = pd.DataFrame(results)
        results_df = results_df.sort_values("simple_mean_auc", ascending=False, na_position="last")

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        results_df.to_csv(output_path, index=False)
        log.info(f"\nSaved {len(results_df)} results to {output_path}")

        # Save tree rules to separate file
        rules_path = Path(args.output).with_suffix(".trees.txt")
        with open(rules_path, "w") as f:
            for _, row in results_df.iterrows():
                if pd.notna(row.get("tree_rules")) and row["tree_rules"]:
                    f.write(f"=== {row['endpoint']} (AUC={row['adapter_tree_auc']}) ===\n")
                    f.write(row["tree_rules"])
                    f.write("\n\n")
        log.info(f"Saved tree rules to {rules_path}")

        # Drop verbose text columns from CSV
        csv_df = results_df.drop(columns=["tree_rules", "logreg_desc"], errors="ignore")
        csv_df.to_csv(output_path, index=False)

        # Save logreg descriptions to separate file
        logreg_path = Path(args.output).with_suffix(".logreg.txt")
        with open(logreg_path, "w") as f:
            for _, row in results_df.iterrows():
                if pd.notna(row.get("logreg_desc")) and row["logreg_desc"]:
                    f.write(f"=== {row['endpoint']} (AUC={row['adapter_logreg_auc']}) ===\n")
                    f.write(row["logreg_desc"])
                    f.write("\n\n")
        log.info(f"Saved logreg descriptions to {logreg_path}")

        # Print summary
        print("\n" + "=" * 130)
        print("LINKAGE EVALUATION SUMMARY")
        print("=" * 130)

        print(f"\n{'Endpoint':<50} {'Conf':>6} {'N':>5} {'Tok':>4} {'Mean':>7} {'Tree':>7} {'LogReg':>7} {'FP':>7} {'LR-FP':>7}")
        print("-" * 130)

        for _, row in results_df.iterrows():
            ep = row["endpoint"][:49]
            conf = row["confidence"][:6]
            n = row["n_evaluated"]
            nt = row["n_tokens"]
            sm = f"{row['simple_mean_auc']:.3f}" if pd.notna(row["simple_mean_auc"]) else "  -"
            tr = f"{row['adapter_tree_auc']:.3f}" if pd.notna(row["adapter_tree_auc"]) else "  -"
            lr = f"{row['adapter_logreg_auc']:.3f}" if pd.notna(row.get("adapter_logreg_auc")) else "  -"
            fp = f"{row['fingerprint_auc']:.3f}" if pd.notna(row["fingerprint_auc"]) else "  -"
            dvfp = f"{row['delta_logreg_vs_fp']:+.3f}" if pd.notna(row.get("delta_logreg_vs_fp")) else "     -"
            print(f"{ep:<50} {conf:>6} {n:>5} {nt:>4} {sm:>7} {tr:>7} {lr:>7} {fp:>7} {dvfp:>7}")

        # Aggregate stats
        valid_sm = results_df["simple_mean_auc"].dropna()
        valid_tree = results_df["adapter_tree_auc"].dropna()
        valid_lr = results_df["adapter_logreg_auc"].dropna()
        valid_fp = results_df["fingerprint_auc"].dropna()
        valid_delta_fp = results_df["delta_logreg_vs_fp"].dropna()

        print(f"\n{'Metric':<35} {'Value':>10}")
        print("-" * 47)
        print(f"{'Endpoints evaluated':<35} {len(results_df):>10}")
        print(f"{'With linked tokens':<35} {(results_df['n_tokens'] > 0).sum():>10}")
        print(f"{'Skipped (no tokens)':<35} {skipped_none:>10}")

        if len(valid_sm) > 0:
            print(f"{'Mean simple-mean AUC':<35} {valid_sm.mean():>10.4f}")
            print(f"{'Median simple-mean AUC':<35} {valid_sm.median():>10.4f}")
            print(f"{'Simple-mean AUC >= 0.70':<35} {(valid_sm >= 0.70).sum():>10}")

        if len(valid_tree) > 0:
            print(f"{'Mean adapter tree AUC':<35} {valid_tree.mean():>10.4f}")
            print(f"{'Median adapter tree AUC':<35} {valid_tree.median():>10.4f}")

        if len(valid_lr) > 0:
            print(f"{'Mean adapter logreg AUC':<35} {valid_lr.mean():>10.4f}")
            print(f"{'Median adapter logreg AUC':<35} {valid_lr.median():>10.4f}")
            print(f"{'Adapter logreg AUC >= 0.70':<35} {(valid_lr >= 0.70).sum():>10}")

        if len(valid_fp) > 0:
            print(f"{'Mean fingerprint AUC':<35} {valid_fp.mean():>10.4f}")
            print(f"{'Median fingerprint AUC':<35} {valid_fp.median():>10.4f}")

        if len(valid_delta_fp) > 0:
            print(f"{'Mean delta (logreg - FP)':<35} {valid_delta_fp.mean():>+10.4f}")
            n_beat_fp = (valid_delta_fp > 0).sum()
            n_lost_fp = (valid_delta_fp < 0).sum()
            n_total = n_beat_fp + n_lost_fp
            print(f"{'Logreg beats fingerprint':<35} {n_beat_fp:>10} / {n_total}")
            print(f"{'Fingerprint beats logreg':<35} {n_lost_fp:>10} / {n_total}")

        # Breakdown by confidence
        print(f"\nBy confidence level:")
        for conf in ["high", "medium", "low", "none"]:
            subset = results_df[results_df["confidence"] == conf]
            if len(subset) > 0:
                sm_vals = subset["simple_mean_auc"].dropna()
                lr_vals = subset["adapter_logreg_auc"].dropna()
                sm_str = f"mean={sm_vals.mean():.3f}" if len(sm_vals) > 0 else "mean=N/A"
                lr_str = f"logreg={lr_vals.mean():.3f}" if len(lr_vals) > 0 else "logreg=N/A"
                print(f"  {conf}: {len(subset)} endpoints, {sm_str}, {lr_str}")
    else:
        log.warning("No results to save")


if __name__ == "__main__":
    main()
