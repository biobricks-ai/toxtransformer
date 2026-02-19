#!/usr/bin/env python3
"""
Experiment: Full 6,647-feature adapter vs linked-token adapter vs Morgan FP.

Tests whether using ALL ToxTransformer predictions as features in a LogReg
can beat Morgan fingerprints on ADME endpoints where linked tokens (1-4 features) lose.

Also tests an intermediate approach: ~170 ADME-relevant tokens only.
"""

import json
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolFromInchi
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

MAX_SAMPLES_PER_CLASS = 250

# ADME endpoints to test (the ones where FP currently wins)
ADME_ENDPOINTS = [
    "cyp450||cyp450/cyp450_cyp1a2_combined/label",
    "cyp450||cyp450/cyp450_cyp2c9_combined/label",
    "cyp450||cyp450/cyp450_cyp2c19_combined/label",
    "cyp450||cyp450/cyp450_cyp2d6_combined/label",
    "cyp450||cyp450/cyp450_cyp3a4_combined/label",
    "pharmabench||pharmabench/pharmabench_bbb_penetration/value",
]

# Also test some toxicity endpoints as controls
TOXICITY_CONTROLS = [
    "ames-benchmark||ames-benchmark/ames_cv_splits/ames_class",
    "admet-huggingface||admet-huggingface/admet/hERG",
    "tdc||tdc/tdc_dili/label",
]


def load_feature_cache(cache_paths):
    """Load all cached features into memory."""
    feature_cache = {}
    property_tokens = None

    for cache_path in cache_paths:
        if not Path(cache_path).exists():
            continue
        conn = sqlite3.connect(cache_path)
        cursor = conn.execute("SELECT inchi, features, property_tokens FROM feature_cache")
        n = 0
        for row in cursor:
            inchi = row[0]
            if inchi not in feature_cache:
                features = np.frombuffer(row[1], dtype=np.float32)
                feature_cache[inchi] = features
                n += 1
                if property_tokens is None:
                    property_tokens = json.loads(row[2])
        conn.close()
        print(f"  Loaded {n} features from {cache_path}")

    print(f"  Total: {len(feature_cache)} molecules, {len(property_tokens)} properties")
    return feature_cache, property_tokens


def get_adme_token_indices(property_tokens, sqlite_path):
    """Get indices of ADME-relevant tokens from the SQLite catalog."""
    conn = sqlite3.connect(sqlite_path)

    # Get all CYP, BBB, transporter, PK tokens
    query = """
    SELECT DISTINCT p.property_token, p.title, s.source
    FROM property p
    JOIN source s ON p.source_id = s.source_id
    WHERE (
        p.title LIKE '%CYP%' OR p.title LIKE '%cytochrome%'
        OR p.title LIKE '%BBB%' OR p.title LIKE '%blood-brain%'
        OR p.title LIKE '%P-glycoprotein%' OR p.title LIKE '%ABCB%'
        OR p.title LIKE '%ABCC%' OR p.title LIKE '%ABCG%'
        OR p.title LIKE '%MDR%' OR p.title LIKE '%Pgp%'
        OR p.title LIKE '%hepatic clearance%' OR p.title LIKE '%plasma%unbound%'
        OR p.title LIKE '%solubility%' OR p.title LIKE '%PAMPA%'
        OR p.title LIKE '%microsom%' OR p.title LIKE '%SULT%'
        OR p.title LIKE '%UGT%' OR p.title LIKE '%glucuronid%'
    )
    AND p.property_token IS NOT NULL
    """
    rows = conn.execute(query).fetchall()
    conn.close()

    adme_tokens = set()
    for token, title, source in rows:
        if token is not None:
            adme_tokens.add(int(token))

    # Map to indices
    token_to_idx = {t: i for i, t in enumerate(property_tokens)}
    adme_indices = [token_to_idx[t] for t in adme_tokens if t in token_to_idx]

    print(f"  Found {len(adme_tokens)} ADME tokens, {len(adme_indices)} mapped to feature indices")
    return sorted(adme_indices), sorted(adme_tokens)


def balanced_sample(df, max_per_class=MAX_SAMPLES_PER_CLASS, random_state=42):
    pos = df[df["value"] == 1]
    neg = df[df["value"] == 0]
    n = min(max_per_class, len(pos), len(neg))
    if n < 5:
        return pd.DataFrame()
    return pd.concat([
        pos.sample(n=n, random_state=random_state),
        neg.sample(n=n, random_state=random_state),
    ])


def evaluate_logreg(X, y, C=1.0, n_folds=5, random_state=42):
    """5-fold CV LogReg, returns mean AUC."""
    if len(y) < 20 or len(set(y)) < 2:
        return None
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if min(n_pos, n_neg) < n_folds:
        return None

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_aucs = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(set(y_test)) < 2:
            continue
        clf = LogisticRegression(
            max_iter=2000, C=C, random_state=random_state,
            solver="lbfgs", penalty="l2",
        )
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        fold_aucs.append(roc_auc_score(y_test, y_prob))

    return float(np.mean(fold_aucs)) if fold_aucs else None


def evaluate_logreg_tuned(X, y, n_folds=5, random_state=42):
    """5-fold CV LogReg with C tuned via inner CV on training folds."""
    C_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

    if len(y) < 20 or len(set(y)) < 2:
        return None, None
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if min(n_pos, n_neg) < n_folds:
        return None, None

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    fold_aucs = []
    best_Cs = []

    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        if len(set(y_test)) < 2:
            continue

        # Inner CV to pick C
        best_C = 1.0
        best_inner_auc = 0
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=random_state)

        for C in C_values:
            inner_aucs = []
            for itrain, ival in inner_skf.split(X_train, y_train):
                if len(set(y_train[ival])) < 2:
                    continue
                clf = LogisticRegression(
                    max_iter=2000, C=C, random_state=random_state,
                    solver="lbfgs", penalty="l2",
                )
                clf.fit(X_train[itrain], y_train[itrain])
                try:
                    inner_aucs.append(roc_auc_score(y_train[ival], clf.predict_proba(X_train[ival])[:, 1]))
                except ValueError:
                    pass
            if inner_aucs and np.mean(inner_aucs) > best_inner_auc:
                best_inner_auc = np.mean(inner_aucs)
                best_C = C

        best_Cs.append(best_C)

        # Train with best C on full training fold
        clf = LogisticRegression(
            max_iter=2000, C=best_C, random_state=random_state,
            solver="lbfgs", penalty="l2",
        )
        clf.fit(X_train, y_train)
        y_prob = clf.predict_proba(X_test)[:, 1]
        fold_aucs.append(roc_auc_score(y_test, y_prob))

    if not fold_aucs:
        return None, None

    # Most common C
    from collections import Counter
    most_common_C = Counter(best_Cs).most_common(1)[0][0]
    return float(np.mean(fold_aucs)), most_common_C


def compute_fingerprints(inchis):
    """Convert InChIs to Morgan FP matrix."""
    fps = []
    mask = []
    for inchi in inchis:
        mol = MolFromInchi(inchi, sanitize=True, removeHs=True)
        if mol is not None:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fps.append(np.array(fp, dtype=np.float32))
            mask.append(True)
        else:
            mask.append(False)
    return np.vstack(fps) if fps else None, np.array(mask)


def main():
    print("=" * 100)
    print("EXPERIMENT: Full 6,647-Feature Adapter vs Linked Tokens vs Morgan FP")
    print("=" * 100)

    # Load data
    print("\nLoading feature cache...")
    cache_paths = [
        "cache/adapter_all_eval_features.sqlite",
        "cache/adapter_eval_features.sqlite",
    ]
    feature_cache, property_tokens = load_feature_cache(cache_paths)

    print("\nLoading ADME token catalog...")
    adme_indices, adme_tokens = get_adme_token_indices(
        property_tokens, "cache/build_sqlite/cvae.sqlite"
    )

    print("\nLoading linkages...")
    with open("cache/property_linkages.json") as f:
        linkages = json.load(f)

    print("\nLoading benchmark data...")
    benchmark_df = pd.read_parquet("publication/benchmark/data/external_binary_benchmark.parquet")
    print(f"  {len(benchmark_df):,} records")

    token_to_idx = {t: i for i, t in enumerate(property_tokens)}

    # Results table
    all_endpoints = ADME_ENDPOINTS + TOXICITY_CONTROLS
    results = []

    for endpoint_key in all_endpoints:
        is_adme = endpoint_key in ADME_ENDPOINTS
        linkage = linkages.get(endpoint_key)
        if not linkage:
            print(f"\n  SKIP {endpoint_key}: not in linkages")
            continue

        source = linkage["source"]
        prop = linkage["property"]
        linked_tokens = linkage.get("tokens", [])

        # Get data
        mask = (benchmark_df["source"] == source) & (benchmark_df["property"] == prop)
        prop_df = benchmark_df[mask]

        # Filter to cached
        prop_df = prop_df[prop_df["inchi"].isin(feature_cache)]
        sampled = balanced_sample(prop_df)
        if len(sampled) < 20:
            print(f"\n  SKIP {endpoint_key}: insufficient samples ({len(sampled)})")
            continue

        inchis = sampled["inchi"].tolist()
        y = sampled["value"].values.astype(int)
        n_pos = int(y.sum())

        print(f"\n{'ADME' if is_adme else 'TOX '} | {endpoint_key}")
        print(f"  N={len(sampled)}, pos={n_pos}, linked_tokens={len(linked_tokens)}")

        # Build full feature matrix
        X_full = np.vstack([feature_cache[inchi] for inchi in inchis])

        # 1. Linked tokens only (1-4 features), C=1.0 (current approach)
        linked_indices = [token_to_idx[t] for t in linked_tokens if t in token_to_idx]
        X_linked = X_full[:, linked_indices]
        auc_linked = evaluate_logreg(X_linked, y, C=1.0)

        # 2. Full 6,647 features, C=1.0
        auc_full_c1 = evaluate_logreg(X_full, y, C=1.0)

        # 3. Full 6,647 features, tuned C (inner CV)
        auc_full_tuned, best_C = evaluate_logreg_tuned(X_full, y)

        # 4. ADME-only features (~170), tuned C
        X_adme = X_full[:, adme_indices]
        auc_adme_tuned, adme_C = evaluate_logreg_tuned(X_adme, y)

        # 5. Morgan fingerprints, C=1.0 (existing baseline)
        X_fp, fp_mask = compute_fingerprints(inchis)
        auc_fp = None
        auc_fp_tuned = None
        if X_fp is not None and fp_mask.sum() >= 20:
            y_fp = y[fp_mask]
            auc_fp = evaluate_logreg(X_fp, y_fp, C=1.0)
            # Also tune FP for fairness
            auc_fp_tuned, fp_C = evaluate_logreg_tuned(X_fp, y_fp)

        row = {
            "endpoint": endpoint_key.split("||")[-1].split("/")[-1],
            "type": "ADME" if is_adme else "TOX",
            "N": len(sampled),
            "n_linked": len(linked_indices),
            "linked_C1": auc_linked,
            "full_C1": auc_full_c1,
            "full_tuned": auc_full_tuned,
            "full_best_C": best_C,
            "adme_tuned": auc_adme_tuned,
            "adme_best_C": adme_C,
            "fp_C1": auc_fp,
            "fp_tuned": auc_fp_tuned,
        }
        results.append(row)

        def fmt(v):
            return f"{v:.3f}" if v is not None else "  N/A"

        print(f"  Linked ({len(linked_indices)}feat, C=1):  {fmt(auc_linked)}")
        print(f"  Full (6647feat, C=1):    {fmt(auc_full_c1)}")
        print(f"  Full (6647feat, tuned):  {fmt(auc_full_tuned)}  (C={best_C})")
        print(f"  ADME ({len(adme_indices)}feat, tuned): {fmt(auc_adme_tuned)}  (C={adme_C})")
        print(f"  FP (2048feat, C=1):      {fmt(auc_fp)}")
        print(f"  FP (2048feat, tuned):    {fmt(auc_fp_tuned)}")

    # Summary table
    print("\n" + "=" * 130)
    print("RESULTS SUMMARY")
    print("=" * 130)
    print(f"\n{'Endpoint':<25} {'Type':<5} {'N':>4} {'Linked':>8} {'Full C=1':>8} {'Full Tune':>9} {'ADME Tune':>9} {'FP C=1':>8} {'FP Tune':>8} {'Winner':>12}")
    print("-" * 130)

    for r in results:
        def fmt(v):
            return f"{v:.3f}" if v is not None else "   N/A"

        # Determine winner among full_tuned vs fp_tuned
        candidates = {}
        if r["full_tuned"] is not None:
            candidates["FullTT"] = r["full_tuned"]
        if r["adme_tuned"] is not None:
            candidates["ADME-TT"] = r["adme_tuned"]
        if r["fp_tuned"] is not None:
            candidates["FP"] = r["fp_tuned"]

        winner = max(candidates, key=candidates.get) if candidates else "N/A"
        winner_val = candidates.get(winner, 0)

        print(f"{r['endpoint']:<25} {r['type']:<5} {r['N']:>4} "
              f"{fmt(r['linked_C1']):>8} {fmt(r['full_C1']):>8} {fmt(r['full_tuned']):>9} "
              f"{fmt(r['adme_tuned']):>9} {fmt(r['fp_C1']):>8} {fmt(r['fp_tuned']):>8} "
              f"{winner:>12}")

    # Head-to-head: Full TT tuned vs FP tuned on ADME endpoints
    print("\n\nHEAD-TO-HEAD: Full TT (tuned) vs FP (tuned) on ADME endpoints:")
    tt_wins = 0
    fp_wins = 0
    for r in results:
        if r["type"] != "ADME":
            continue
        if r["full_tuned"] is not None and r["fp_tuned"] is not None:
            delta = r["full_tuned"] - r["fp_tuned"]
            winner = "TT" if delta > 0 else "FP"
            if delta > 0:
                tt_wins += 1
            else:
                fp_wins += 1
            print(f"  {r['endpoint']:<25}: TT={r['full_tuned']:.3f}  FP={r['fp_tuned']:.3f}  delta={delta:+.3f}  [{winner}]")
    print(f"\n  TT wins: {tt_wins}, FP wins: {fp_wins}")


if __name__ == "__main__":
    main()
