#!/usr/bin/env python3
"""
Experiment v2: TT+FP combined features, elastic net, and feature selection.

Building on v1 findings:
- Full 6,647 TT features close the gap but don't consistently beat FP
- Now test: (a) combining TT+FP, (b) elastic net, (c) feature importance analysis
"""

import json
import sqlite3
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.inchi import MolFromInchi
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=FutureWarning)

MAX_SAMPLES_PER_CLASS = 250

ENDPOINTS = {
    "CYP1A2": "cyp450||cyp450/cyp450_cyp1a2_combined/label",
    "CYP2C9": "cyp450||cyp450/cyp450_cyp2c9_combined/label",
    "CYP2C19": "cyp450||cyp450/cyp450_cyp2c19_combined/label",
    "CYP2D6": "cyp450||cyp450/cyp450_cyp2d6_combined/label",
    "CYP3A4": "cyp450||cyp450/cyp450_cyp3a4_combined/label",
    "BBB": "pharmabench||pharmabench/pharmabench_bbb_penetration/value",
    "hERG": "admet-huggingface||admet-huggingface/admet/hERG",
    "Ames": "ames-benchmark||ames-benchmark/ames_cv_splits/ames_class",
}


def load_feature_cache(cache_paths):
    feature_cache = {}
    property_tokens = None
    for cache_path in cache_paths:
        if not Path(cache_path).exists():
            continue
        conn = sqlite3.connect(cache_path)
        cursor = conn.execute("SELECT inchi, features, property_tokens FROM feature_cache")
        for row in cursor:
            inchi = row[0]
            if inchi not in feature_cache:
                feature_cache[inchi] = np.frombuffer(row[1], dtype=np.float32)
                if property_tokens is None:
                    property_tokens = json.loads(row[2])
        conn.close()
    return feature_cache, property_tokens


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


def cv_evaluate(X, y, C=1.0, penalty="l2", l1_ratio=None, scale=False, n_folds=5, seed=42):
    """5-fold CV with optional scaling and elastic net."""
    if len(y) < 20 or len(set(y)) < 2:
        return None
    if min(int(y.sum()), len(y) - int(y.sum())) < n_folds:
        return None

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aucs = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(set(y_te)) < 2:
            continue

        if scale:
            scaler = StandardScaler()
            X_tr = scaler.fit_transform(X_tr)
            X_te = scaler.transform(X_te)

        if penalty == "elasticnet":
            clf = SGDClassifier(
                loss="log_loss", penalty="elasticnet",
                alpha=1.0 / (C * len(y_tr)), l1_ratio=l1_ratio or 0.5,
                max_iter=2000, random_state=seed,
            )
        else:
            clf = LogisticRegression(
                max_iter=2000, C=C, random_state=seed,
                solver="lbfgs", penalty=penalty,
            )
        clf.fit(X_tr, y_tr)

        if hasattr(clf, "predict_proba"):
            y_prob = clf.predict_proba(X_te)[:, 1]
        else:
            y_prob = clf.decision_function(X_te)

        fold_aucs.append(roc_auc_score(y_te, y_prob))

    return float(np.mean(fold_aucs)) if fold_aucs else None


def cv_evaluate_tuned(X, y, penalty="l2", scale=False, n_folds=5, seed=42):
    """5-fold CV with inner CV for C tuning."""
    C_values = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
    if len(y) < 20 or len(set(y)) < 2:
        return None, None
    if min(int(y.sum()), len(y) - int(y.sum())) < n_folds:
        return None, None

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    fold_aucs = []
    best_Cs = []

    for train_idx, test_idx in skf.split(X, y):
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        if len(set(y_te)) < 2:
            continue

        if scale:
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)
        else:
            X_tr_s, X_te_s = X_tr, X_te

        # Inner CV for C
        best_C, best_inner = 1.0, 0
        inner_skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)
        for C in C_values:
            inner_aucs = []
            for itr, ival in inner_skf.split(X_tr_s, y_tr):
                if len(set(y_tr[ival])) < 2:
                    continue
                clf = LogisticRegression(
                    max_iter=2000, C=C, random_state=seed,
                    solver="lbfgs", penalty=penalty,
                )
                clf.fit(X_tr_s[itr], y_tr[itr])
                try:
                    inner_aucs.append(roc_auc_score(y_tr[ival], clf.predict_proba(X_tr_s[ival])[:, 1]))
                except ValueError:
                    pass
            if inner_aucs and np.mean(inner_aucs) > best_inner:
                best_inner = np.mean(inner_aucs)
                best_C = C

        best_Cs.append(best_C)
        clf = LogisticRegression(
            max_iter=2000, C=best_C, random_state=seed,
            solver="lbfgs", penalty=penalty,
        )
        clf.fit(X_tr_s, y_tr)
        fold_aucs.append(roc_auc_score(y_te, clf.predict_proba(X_te_s)[:, 1]))

    if not fold_aucs:
        return None, None
    return float(np.mean(fold_aucs)), Counter(best_Cs).most_common(1)[0][0]


def compute_fps(inchis):
    fps, mask = [], []
    for inchi in inchis:
        mol = MolFromInchi(inchi, sanitize=True, removeHs=True)
        if mol:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
            fps.append(np.array(fp, dtype=np.float32))
            mask.append(True)
        else:
            mask.append(False)
    return (np.vstack(fps) if fps else None), np.array(mask)


def get_top_features(X, y, property_tokens, top_n=50):
    """Quick feature importance via univariate AUC."""
    aucs = []
    for j in range(X.shape[1]):
        try:
            a = roc_auc_score(y, X[:, j])
            aucs.append(max(a, 1 - a))  # Handle inverse correlations
        except ValueError:
            aucs.append(0.5)
    aucs = np.array(aucs)
    top_idx = np.argsort(aucs)[-top_n:]
    return top_idx, aucs[top_idx], [property_tokens[i] for i in top_idx]


def main():
    print("=" * 120)
    print("EXPERIMENT v2: Combined TT+FP, Feature Selection, Scaling")
    print("=" * 120)

    cache_paths = [
        "cache/adapter_all_eval_features.sqlite",
        "cache/adapter_eval_features.sqlite",
    ]
    feature_cache, property_tokens = load_feature_cache(cache_paths)
    print(f"Loaded {len(feature_cache)} molecules, {len(property_tokens)} properties")

    with open("cache/property_linkages.json") as f:
        linkages = json.load(f)
    benchmark_df = pd.read_parquet("publication/benchmark/data/external_binary_benchmark.parquet")

    token_to_idx = {t: i for i, t in enumerate(property_tokens)}

    results = []

    for name, endpoint_key in ENDPOINTS.items():
        linkage = linkages.get(endpoint_key)
        if not linkage:
            continue

        source, prop = linkage["source"], linkage["property"]
        linked_tokens = linkage.get("tokens", [])

        mask = (benchmark_df["source"] == source) & (benchmark_df["property"] == prop)
        prop_df = benchmark_df[mask]
        prop_df = prop_df[prop_df["inchi"].isin(feature_cache)]
        sampled = balanced_sample(prop_df)
        if len(sampled) < 20:
            continue

        inchis = sampled["inchi"].tolist()
        y = sampled["value"].values.astype(int)

        # Build feature matrices
        X_tt = np.vstack([feature_cache[inchi] for inchi in inchis])
        X_fp, fp_mask = compute_fps(inchis)

        # For combined, we need both to succeed
        if X_fp is None or fp_mask.sum() < 20:
            continue

        # Use only molecules where FP succeeded (virtually all)
        y_fp = y[fp_mask]
        X_tt_fp = X_tt[fp_mask]

        # Combined: TT + FP
        X_combined = np.hstack([X_tt_fp, X_fp])

        # Linked tokens
        linked_idx = [token_to_idx[t] for t in linked_tokens if t in token_to_idx]
        X_linked = X_tt_fp[:, linked_idx]

        # Top-50 features by univariate AUC
        top_idx, top_aucs, top_tokens = get_top_features(X_tt_fp, y_fp, property_tokens, top_n=50)
        X_top50 = X_tt_fp[:, top_idx]

        # Top-100 features
        top100_idx, _, _ = get_top_features(X_tt_fp, y_fp, property_tokens, top_n=100)
        X_top100 = X_tt_fp[:, top100_idx]

        print(f"\n{'='*80}")
        print(f"{name} (N={len(y_fp)}, linked={len(linked_idx)} tokens)")
        print(f"{'='*80}")

        # Methods to test
        methods = {}

        # 1. Linked tokens, C=1 (current)
        methods["Linked (current)"] = cv_evaluate(X_linked, y_fp, C=1.0)

        # 2. FP, tuned
        auc_fp, fp_C = cv_evaluate_tuned(X_fp, y_fp)
        methods["FP (tuned)"] = auc_fp

        # 3. Full TT, tuned
        auc_tt, tt_C = cv_evaluate_tuned(X_tt_fp, y_fp)
        methods["Full TT (tuned)"] = auc_tt

        # 4. Full TT, tuned + scaled
        auc_tt_s, tt_s_C = cv_evaluate_tuned(X_tt_fp, y_fp, scale=True)
        methods["Full TT (scaled)"] = auc_tt_s

        # 5. Top-50 TT features by univariate AUC, tuned
        auc_t50, t50_C = cv_evaluate_tuned(X_top50, y_fp)
        methods["Top-50 TT (tuned)"] = auc_t50

        # 6. Top-100 TT features, tuned
        auc_t100, t100_C = cv_evaluate_tuned(X_top100, y_fp)
        methods["Top-100 TT (tuned)"] = auc_t100

        # 7. Combined TT+FP, tuned
        auc_comb, comb_C = cv_evaluate_tuned(X_combined, y_fp)
        methods["TT+FP Combined (tuned)"] = auc_comb

        # 8. Combined TT+FP, scaled+tuned
        auc_comb_s, _ = cv_evaluate_tuned(X_combined, y_fp, scale=True)
        methods["TT+FP Combined (scaled)"] = auc_comb_s

        # 9. Linked + FP combined
        X_linked_fp = np.hstack([X_linked, X_fp])
        auc_lfp, _ = cv_evaluate_tuned(X_linked_fp, y_fp)
        methods["Linked+FP (tuned)"] = auc_lfp

        # Print results
        print(f"\n  {'Method':<35} {'AUC':>8}")
        print(f"  {'-'*45}")
        best_method = max(methods, key=lambda k: methods[k] or 0)
        for method, auc in methods.items():
            marker = " <<<" if method == best_method else ""
            print(f"  {method:<35} {auc:.3f}{marker}" if auc else f"  {method:<35}    N/A")

        # Show top-5 most predictive TT features
        print(f"\n  Top-5 univariate TT features:")
        for i in range(min(5, len(top_tokens))):
            idx = -(i + 1)
            token = top_tokens[idx]
            auc_val = top_aucs[idx]
            # Is this token in the linked set?
            linked_marker = " [LINKED]" if token in linked_tokens else ""
            print(f"    Token {token}: univariate AUC={auc_val:.3f}{linked_marker}")

        results.append({
            "endpoint": name,
            "N": len(y_fp),
            "linked": methods.get("Linked (current)"),
            "fp": methods.get("FP (tuned)"),
            "full_tt": methods.get("Full TT (tuned)"),
            "full_tt_scaled": methods.get("Full TT (scaled)"),
            "top50": methods.get("Top-50 TT (tuned)"),
            "top100": methods.get("Top-100 TT (tuned)"),
            "tt_fp": methods.get("TT+FP Combined (tuned)"),
            "tt_fp_scaled": methods.get("TT+FP Combined (scaled)"),
            "linked_fp": methods.get("Linked+FP (tuned)"),
            "best_method": best_method,
        })

    # Final summary
    print("\n" + "=" * 150)
    print("SUMMARY: Best AUC by method across all endpoints")
    print("=" * 150)
    print(f"\n{'Endpoint':<10} {'N':>4} {'Linked':>8} {'FP':>8} {'FullTT':>8} {'TT+Scl':>8} {'Top50':>8} {'Top100':>8} {'TT+FP':>8} {'TT+FP+S':>8} {'Lnk+FP':>8} {'Best':>20}")
    print("-" * 150)

    for r in results:
        def f(v): return f"{v:.3f}" if v is not None else "  N/A"
        print(f"{r['endpoint']:<10} {r['N']:>4} "
              f"{f(r['linked']):>8} {f(r['fp']):>8} {f(r['full_tt']):>8} {f(r['full_tt_scaled']):>8} "
              f"{f(r['top50']):>8} {f(r['top100']):>8} {f(r['tt_fp']):>8} {f(r['tt_fp_scaled']):>8} "
              f"{f(r['linked_fp']):>8} {r['best_method']:>20}")

    # Summary: TT-based method beats FP?
    print("\n\nHEAD-TO-HEAD: Best TT-based method vs Best FP-based method")
    print("-" * 80)
    for r in results:
        tt_methods = {k: v for k, v in r.items()
                      if k in ("linked", "full_tt", "full_tt_scaled", "top50", "top100") and v is not None}
        fp_methods = {k: v for k, v in r.items()
                      if k in ("fp",) and v is not None}
        combined = {k: v for k, v in r.items()
                    if k in ("tt_fp", "tt_fp_scaled", "linked_fp") and v is not None}

        best_tt_name = max(tt_methods, key=tt_methods.get) if tt_methods else "N/A"
        best_tt = max(tt_methods.values()) if tt_methods else 0
        best_fp = max(fp_methods.values()) if fp_methods else 0
        best_combined_name = max(combined, key=combined.get) if combined else "N/A"
        best_combined = max(combined.values()) if combined else 0

        delta_tt = best_tt - best_fp
        delta_comb = best_combined - best_fp

        print(f"  {r['endpoint']:<10}: TT={best_tt:.3f} ({best_tt_name})  FP={best_fp:.3f}  "
              f"Combined={best_combined:.3f} ({best_combined_name})  "
              f"TT-FP={delta_tt:+.3f}  Comb-FP={delta_comb:+.3f}")


if __name__ == "__main__":
    main()
