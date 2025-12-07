"""
Build a table comparing performance across different context sizes.

Columns: Source | No Context | 1-5 Props | 6-10 Props | 11-20 Props
Each cell shows: delta AUC vs prior column, overall AUC, sample count
"""

import pandas as pd
import sqlite3
from sklearn.metrics import roc_auc_score
import numpy as np
import logging
import pathlib

# Setup output directory and logging
outdir = pathlib.Path('cache/build_report/context_performance')
outdir.mkdir(exist_ok=True, parents=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.FileHandler(outdir / 'context_performance.log', mode='w'),
        logging.StreamHandler()
    ]
)


def get_property_source_mapping():
    """Get property token to source mapping from database."""
    conn = sqlite3.connect('cache/build_sqlite/cvae.sqlite')
    prop_source = pd.read_sql('''
        SELECT p.property_token, s.source
        FROM property p
        JOIN source s ON p.source_id = s.source_id
    ''', conn)
    conn.close()
    return prop_source


def calculate_property_aucs(df, group_cols):
    """Calculate AUC for each property within each group, then return median AUC per group."""
    # First calculate AUC per property within each group
    property_aucs = []
    for name, group in df.groupby(group_cols + ['property_token']):
        if len(group['true_value'].unique()) < 2:
            continue
        try:
            auc = roc_auc_score(group['true_value'], group['prob_of_1'])
            if isinstance(name, tuple):
                row = dict(zip(group_cols + ['property_token'], name))
            else:
                row = {group_cols[0]: name, 'property_token': name}
            row['auc'] = auc
            row['n_samples'] = len(group)
            property_aucs.append(row)
        except Exception:
            pass

    if not property_aucs:
        return pd.DataFrame()

    prop_df = pd.DataFrame(property_aucs)

    # Now aggregate to median AUC per group
    results = prop_df.groupby(group_cols).agg(
        median_auc=('auc', 'median'),
        n_properties=('property_token', 'nunique'),
        n_samples=('n_samples', 'sum')
    ).reset_index()

    return results


def main():
    # Get property to source mapping
    prop_source = get_property_source_mapping()

    # Load context evaluation data
    df_context = pd.read_parquet('cache/generate_evaluations_holdout_context/evaluations.parquet')

    # Load no-context data
    df_nprops1 = pd.read_parquet('cache/generate_evaluations_holdout_nprops1/evaluations.parquet')

    # Merge with source mapping
    df_context = df_context.merge(prop_source, on='property_token', how='left')
    df_nprops1 = df_nprops1.merge(prop_source, on='property_token', how='left')

    # Calculate median AUC by source for no-context baseline
    no_context_auc = calculate_property_aucs(df_nprops1, ['source'])
    no_context_auc = no_context_auc.rename(columns={'median_auc': 'no_context_auc', 'n_properties': 'no_context_n'})

    # Calculate median AUC by source and context bucket
    context_auc = calculate_property_aucs(df_context, ['source', 'context_bucket'])

    # Pivot context data
    context_pivot = context_auc.pivot(index='source', columns='context_bucket', values='median_auc')
    context_n_pivot = context_auc.pivot(index='source', columns='context_bucket', values='n_properties')

    # Merge with no-context baseline
    result = no_context_auc.set_index('source').join(context_pivot)
    result_n = no_context_auc.set_index('source')[['no_context_n']].join(context_n_pivot)

    # Reorder columns
    bucket_order = ['1-5', '6-10', '11-20']
    available_buckets = [b for b in bucket_order if b in result.columns]

    # Calculate deltas
    prev_col = 'no_context_auc'
    for bucket in available_buckets:
        result[f'{bucket}_delta'] = result[bucket] - result[prev_col]
        prev_col = bucket

    # Build CSV data
    csv_rows = []

    # Build display table
    logging.info("\nCONTEXT PERFORMANCE BY SOURCE (Median AUC across properties)")
    logging.info("Format: delta / median_AUC / n_properties (delta relative to previous column)\n")

    # Header
    header = f"{'Source':<12}  {'No Context':>22}  {'1-5':>22}  {'6-10':>22}  {'11-20':>22}"
    logging.info(header)
    logging.info("-" * len(header))

    # Format: delta / AUC / n_samples
    for source in sorted(result.index.dropna()):
        row = f"{source:<12}  "
        csv_row = {'source': source}

        # No context column
        auc = result.loc[source, 'no_context_auc']
        n = result_n.loc[source, 'no_context_n'] if source in result_n.index else 0
        row += f"    -- / {auc:.4f} / {int(n):>7}  "
        csv_row['no_context_auc'] = auc
        csv_row['no_context_n'] = int(n)

        # Context columns
        prev_auc = auc
        for bucket in available_buckets:
            if bucket in result.columns and pd.notna(result.loc[source, bucket]):
                bucket_auc = result.loc[source, bucket]
                delta = bucket_auc - prev_auc
                n = result_n.loc[source, bucket] if bucket in result_n.columns else 0
                delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
                row += f"{delta_str} / {bucket_auc:.4f} / {int(n):>7}  "
                csv_row[f'{bucket}_auc'] = bucket_auc
                csv_row[f'{bucket}_delta'] = delta
                csv_row[f'{bucket}_n'] = int(n)
                prev_auc = bucket_auc
            else:
                row += f"{'--':>22}  "

        logging.info(row)
        csv_rows.append(csv_row)

    logging.info("-" * len(header))

    # Overall row - compute median AUC across all properties
    row = f"{'OVERALL':<12}  "
    csv_row = {'source': 'OVERALL'}

    # Overall no-context median AUC
    prop_aucs_no_context = []
    for prop, grp in df_nprops1.groupby('property_token'):
        if len(grp['true_value'].unique()) >= 2:
            try:
                prop_aucs_no_context.append(roc_auc_score(grp['true_value'], grp['prob_of_1']))
            except:
                pass
    overall_no_context = np.median(prop_aucs_no_context) if prop_aucs_no_context else np.nan
    n_props_no_context = len(prop_aucs_no_context)

    row += f"    -- / {overall_no_context:.4f} / {n_props_no_context:>7}  "
    csv_row['no_context_auc'] = overall_no_context
    csv_row['no_context_n'] = n_props_no_context

    # Overall context median AUCs
    prev_auc = overall_no_context
    for bucket in available_buckets:
        bucket_df = df_context[df_context['context_bucket'] == bucket]
        if len(bucket_df) > 0:
            prop_aucs_bucket = []
            for prop, grp in bucket_df.groupby('property_token'):
                if len(grp['true_value'].unique()) >= 2:
                    try:
                        prop_aucs_bucket.append(roc_auc_score(grp['true_value'], grp['prob_of_1']))
                    except:
                        pass
            bucket_auc = np.median(prop_aucs_bucket)
            n_props_bucket = len(prop_aucs_bucket)
            delta = bucket_auc - prev_auc
            delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
            row += f"{delta_str} / {bucket_auc:.4f} / {n_props_bucket:>7}  "
            csv_row[f'{bucket}_auc'] = bucket_auc
            csv_row[f'{bucket}_delta'] = delta
            csv_row[f'{bucket}_n'] = n_props_bucket
            prev_auc = bucket_auc
        else:
            row += f"{'--':>22}  "

    logging.info(row)
    csv_rows.append(csv_row)

    # Save CSV
    csv_df = pd.DataFrame(csv_rows)
    csv_path = outdir / 'context_performance.csv'
    csv_df.to_csv(csv_path, index=False)
    logging.info(f"\nCSV saved to: {csv_path}")


if __name__ == '__main__':
    main()
