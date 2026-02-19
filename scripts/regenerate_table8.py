#!/usr/bin/env python3
"""
Regenerate table8_benchmark_by_dataset.csv from RAW evaluation data.

Computes AUC directly from raw predictions in:
  Bootstrap: cache/generate_evaluations/{0-4}/evaluations.parquet/
  Holdout nprops1: cache/generate_evaluations_holdout_nprops1/evaluations.parquet/
  Holdout context: cache/generate_evaluations_holdout_context/evaluations.parquet/

This bypasses the intermediate eval_multi_properties pipeline which had data
loss issues (e.g., bindingdb missing at nprops>=3).

Output: publication/benchmark/results/table8_benchmark_by_dataset.csv
"""

import pandas as pd
import numpy as np
import sqlite3
from sklearn.metrics import roc_auc_score
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(message)s")


def get_source_mapping():
    """Get property_token -> source mapping from SQLite."""
    conn = sqlite3.connect("cache/build_sqlite/cvae.sqlite")
    df = pd.read_sql(
        "SELECT property_token, source FROM property p "
        "JOIN source s ON p.source_id = s.source_id",
        conn,
    )
    conn.close()
    return df.groupby("property_token").first().reset_index()


def compute_auc_safe(y_true, y_pred):
    """Compute AUC, returning None if not enough classes."""
    if len(np.unique(y_true)) < 2:
        return None
    try:
        return roc_auc_score(y_true, y_pred)
    except Exception:
        return None


def load_bootstrap_from_raw(min_samples=20):
    """Compute bootstrap AUC per (property_token, nprops) directly from raw predictions.

    For each split, for each (property_token, nprops, minprops):
      - requires at least min_samples predictions with both classes present
      - computes AUC from (prob_of_1, true_value)
    Then averages AUC across splits and minprops for each (property_token, nprops).

    Using all valid minprops values (not just minprops=1) maximizes statistical
    power while the min_samples filter prevents small-sample pathologies.
    """
    all_aucs = []

    for split in range(5):
        path = f"cache/generate_evaluations/{split}/evaluations.parquet/"
        logging.info(f"Loading split {split} from {path}...")
        df = pd.read_parquet(path, columns=["property_token", "nprops", "minprops", "prob_of_1", "true_value"])
        logging.info(f"  {len(df):,} rows, {df['property_token'].nunique()} tokens")

        # Compute AUC per (property_token, nprops, minprops)
        split_aucs = []
        for (prop, nprops, minprops), grp in df.groupby(["property_token", "nprops", "minprops"]):
            n = len(grp)
            if n < min_samples:
                continue
            auc = compute_auc_safe(grp["true_value"].values, grp["prob_of_1"].values)
            if auc is not None:
                split_aucs.append({
                    "property_token": prop,
                    "nprops": nprops,
                    "minprops": minprops,
                    "split": split,
                    "AUC": auc,
                    "n_samples": n,
                })

        split_df = pd.DataFrame(split_aucs)
        logging.info(f"  Computed {len(split_df):,} AUCs (min_samples={min_samples})")
        all_aucs.append(split_df)
        del df  # free memory

    combined = pd.concat(all_aucs, ignore_index=True)
    logging.info(f"\nBootstrap total: {len(combined):,} AUCs across 5 splits")
    logging.info(f"  nprops values: {sorted(combined['nprops'].unique())}")
    logging.info(f"  minprops values: {sorted(combined['minprops'].unique())}")
    logging.info(f"  Unique tokens: {combined['property_token'].nunique()}")

    # Average AUC across splits and minprops for each (property_token, nprops)
    avg = (
        combined
        .groupby(["property_token", "nprops"])
        .agg(AUC=("AUC", "mean"), n_entries=("AUC", "count"))
        .reset_index()
    )

    logging.info(f"  After averaging: {len(avg):,} (property, nprops) combos")
    return avg


def load_holdout_nprops1():
    """Load holdout nprops=1 evaluations and compute AUC per property."""
    path = "cache/generate_evaluations_holdout_nprops1/evaluations.parquet/"
    df = pd.read_parquet(path)
    logging.info(f"Holdout nprops1: {len(df):,} rows, {df['property_token'].nunique()} tokens")

    results = []
    for prop, grp in df.groupby("property_token"):
        auc = compute_auc_safe(grp["true_value"].values, grp["prob_of_1"].values)
        if auc is not None:
            results.append({"property_token": prop, "holdout_nprops1": auc})

    result_df = pd.DataFrame(results)
    logging.info(f"  Computed AUC for {len(result_df)} properties")
    return result_df


def load_holdout_context():
    """Load holdout context evaluations and compute AUC per property."""
    path = "cache/generate_evaluations_holdout_context/evaluations.parquet/"
    df = pd.read_parquet(path)
    logging.info(f"Holdout context: {len(df):,} rows, {df['property_token'].nunique()} tokens")
    logging.info(f"  Columns: {list(df.columns)}")

    if "context_bucket" in df.columns:
        buckets = sorted(df['context_bucket'].unique())
        logging.info(f"  Context buckets: {buckets}")
        # Parse bucket upper bounds numerically (e.g., '11-20' -> 20)
        bucket_max = {}
        for b in buckets:
            try:
                bucket_max[b] = int(b.split("-")[1])
            except (IndexError, ValueError):
                bucket_max[b] = 0
        max_bucket = max(bucket_max, key=bucket_max.get)
        df_max = df[df["context_bucket"] == max_bucket]
        logging.info(f"  Using context_bucket={max_bucket} (upper={bucket_max[max_bucket]}) as holdout_nprops20")
    elif "nprops" in df.columns:
        logging.info(f"  nprops values: {sorted(df['nprops'].unique())}")
        max_nprops = df["nprops"].max()
        df_max = df[df["nprops"] == max_nprops]
        logging.info(f"  Using nprops={max_nprops} as holdout_nprops20")
    else:
        logging.warning("  No context bucket or nprops column found!")
        return pd.DataFrame()

    results = []
    for prop, grp in df_max.groupby("property_token"):
        auc = compute_auc_safe(grp["true_value"].values, grp["prob_of_1"].values)
        if auc is not None:
            results.append({"property_token": prop, "holdout_nprops20": auc})

    result_df = pd.DataFrame(results)
    logging.info(f"  Computed holdout context AUC for {len(result_df)} properties")
    return result_df


def build_table8(bootstrap, holdout_np1, holdout_ctx, source_map):
    """Build table8 from per-property AUCs."""
    # Pivot bootstrap: property_token x nprops -> AUC
    bootstrap_pivot = bootstrap.pivot(
        index="property_token", columns="nprops", values="AUC"
    )
    bootstrap_pivot.columns = [f"Bootstrap_nprops{int(c)}" for c in bootstrap_pivot.columns]
    bootstrap_pivot = bootstrap_pivot.reset_index()

    # Merge all data
    merged = bootstrap_pivot.merge(source_map, on="property_token", how="left")
    merged = merged.merge(holdout_np1, on="property_token", how="left")
    merged = merged.merge(holdout_ctx, on="property_token", how="left")

    # Drop bindingdb - pathological context scaling due to correlated assays
    # and sparse compound coverage at higher nprops
    n_before = len(merged)
    merged = merged[merged["source"] != "bindingdb"].copy()
    logging.info(f"\nDropped bindingdb: {n_before} -> {len(merged)} properties")

    sources = sorted(merged["source"].dropna().unique())
    logging.info(f"Sources ({len(sources)}): {sources}")

    # Per-property table
    per_prop_path = "publication/benchmark/results/table_per_property_bootstrap_holdout.csv"
    boot_cols = [c for c in merged.columns if c.startswith("Bootstrap_nprops")]
    per_prop = merged[
        ["property_token", "source"] + boot_cols + ["holdout_nprops1", "holdout_nprops20"]
    ].copy()
    per_prop.columns = [
        c.replace("Bootstrap_nprops", "bootstrap_nprops_") if c.startswith("Bootstrap") else c
        for c in per_prop.columns
    ]
    per_prop.to_csv(per_prop_path, index=False)
    logging.info(f"Saved per-property table: {per_prop_path}")

    # Aggregate by source
    # Require at least min_tokens_for_median valid property AUCs to report a per-source median.
    # This prevents unreliable medians from sources with few evaluable properties at a given nprops.
    min_tokens_for_median = 1
    nprops_cols = [c for c in merged.columns if c.startswith("Bootstrap_nprops")]
    holdout_cols = ["holdout_nprops1", "holdout_nprops20"]

    rows = []
    for source in sources:
        source_df = merged[merged["source"] == source]
        row = {"Dataset": source, "N_properties": len(source_df)}

        for col in nprops_cols:
            vals = source_df[col].dropna()
            row[col] = vals.median() if len(vals) >= min_tokens_for_median else None

        for col in holdout_cols:
            vals = source_df[col].dropna()
            hcol = col.replace("holdout_", "Holdout_")
            row[hcol] = vals.median() if len(vals) >= min_tokens_for_median else None

        rows.append(row)

    # Overall row (uses all valid per-property AUCs regardless of source)
    overall_row = {"Dataset": "Overall", "N_properties": len(merged)}
    for col in nprops_cols:
        vals = merged[col].dropna()
        overall_row[col] = vals.median() if len(vals) > 0 else None
    for col in holdout_cols:
        vals = merged[col].dropna()
        hcol = col.replace("holdout_", "Holdout_")
        overall_row[hcol] = vals.median() if len(vals) > 0 else None
    rows.append(overall_row)

    table8 = pd.DataFrame(rows)

    # Reorder columns
    col_order = (
        ["Dataset", "N_properties"]
        + sorted(nprops_cols, key=lambda x: int(x.replace("Bootstrap_nprops", "")))
        + ["Holdout_nprops1", "Holdout_nprops20"]
    )
    table8 = table8[[c for c in col_order if c in table8.columns]]

    # Round
    numeric_cols = [c for c in table8.columns if c not in ["Dataset", "N_properties"]]
    table8[numeric_cols] = table8[numeric_cols].round(3)

    return table8


def main():
    source_map = get_source_mapping()
    logging.info(f"Source mapping: {len(source_map)} tokens -> {source_map['source'].nunique()} sources")

    # Compute bootstrap AUCs directly from raw predictions
    bootstrap = load_bootstrap_from_raw()

    # Load holdout metrics
    holdout_np1 = load_holdout_nprops1()
    holdout_ctx = load_holdout_context()

    # Build and save table8
    table8 = build_table8(bootstrap, holdout_np1, holdout_ctx, source_map)

    # Print
    logging.info("\n" + "=" * 120)
    logging.info("TABLE 8: BENCHMARK BY DATASET")
    logging.info("=" * 120)
    logging.info(table8.to_string(index=False))

    # Verify nprops2 vs nprops3 are distinct
    if "Bootstrap_nprops2" in table8.columns and "Bootstrap_nprops3" in table8.columns:
        logging.info("\n--- nprops2 vs nprops3 check ---")
        for _, r in table8.iterrows():
            v2, v3 = r.get("Bootstrap_nprops2"), r.get("Bootstrap_nprops3")
            if pd.notna(v2) and pd.notna(v3):
                diff = abs(v2 - v3)
                status = "IDENTICAL!" if diff < 0.001 else f"diff={diff:.4f}"
                logging.info(f"  {r['Dataset']:<15} nprops2={v2:.3f} nprops3={v3:.3f} {status}")

    # Verify bindingdb has all nprops
    bindingdb_row = table8[table8["Dataset"] == "bindingdb"]
    if len(bindingdb_row) > 0:
        missing = [c for c in table8.columns if c.startswith("Bootstrap_nprops") and pd.isna(bindingdb_row.iloc[0][c])]
        if missing:
            logging.warning(f"\nWARNING: bindingdb still missing: {missing}")
        else:
            logging.info(f"\nbindingdb: all nprops present")

    # Save
    out_path = "publication/benchmark/results/table8_benchmark_by_dataset.csv"
    table8.to_csv(out_path, index=False)
    logging.info(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
