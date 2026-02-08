#!/usr/bin/env python3
"""
Generate publication figures from ToxTransformer benchmark evaluation results.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
from pathlib import Path

# Configuration
RESULTS_DIR = Path("publication/benchmark/results")
FIGURES_DIR = Path("publication/benchmark/figures")
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Style settings
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

def load_results():
    """Load evaluation results."""
    results = {}

    # Try to load binary results
    binary_file = RESULTS_DIR / "binary_classification_summary.csv"
    if binary_file.exists():
        results['binary'] = pd.read_csv(binary_file)
        print(f"Loaded binary results: {len(results['binary'])} evaluations")

    # Try to load continuous results
    continuous_file = RESULTS_DIR / "continuous_regression_summary.csv"
    if continuous_file.exists():
        results['continuous'] = pd.read_csv(continuous_file)
        print(f"Loaded continuous results: {len(results['continuous'])} evaluations")

    # Try to load summary
    summary_file = RESULTS_DIR / "evaluation_summary.json"
    if summary_file.exists():
        with open(summary_file) as f:
            results['summary'] = json.load(f)
        print(f"Loaded evaluation summary")

    return results

def plot_binary_performance_by_source(binary_df):
    """Plot binary classification performance by data source."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Aggregate by source
    source_agg = binary_df.groupby('source').agg({
        'accuracy': ['mean', 'std', 'count'],
        'f1': ['mean', 'std'],
        'auc': ['mean', 'std']
    }).reset_index()
    source_agg.columns = ['source', 'acc_mean', 'acc_std', 'n_evals', 'f1_mean', 'f1_std', 'auc_mean', 'auc_std']
    source_agg = source_agg.sort_values('acc_mean', ascending=True)

    # Accuracy plot
    ax = axes[0]
    y_pos = range(len(source_agg))
    ax.barh(y_pos, source_agg['acc_mean'], xerr=source_agg['acc_std'], capsize=3, color='steelblue', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(source_agg['source'], fontsize=8)
    ax.set_xlabel('Accuracy')
    ax.set_title('Accuracy by Source')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5, label='Random')
    ax.set_xlim(0, 1)

    # F1 plot
    ax = axes[1]
    ax.barh(y_pos, source_agg['f1_mean'], xerr=source_agg['f1_std'], capsize=3, color='darkorange', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(source_agg['source'], fontsize=8)
    ax.set_xlabel('F1 Score')
    ax.set_title('F1 Score by Source')
    ax.set_xlim(0, 1)

    # AUC plot
    ax = axes[2]
    auc_data = source_agg.dropna(subset=['auc_mean'])
    y_pos_auc = range(len(auc_data))
    ax.barh(y_pos_auc, auc_data['auc_mean'], xerr=auc_data['auc_std'], capsize=3, color='forestgreen', alpha=0.8)
    ax.set_yticks(y_pos_auc)
    ax.set_yticklabels(auc_data['source'], fontsize=8)
    ax.set_xlabel('AUC-ROC')
    ax.set_title('AUC-ROC by Source')
    ax.axvline(x=0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlim(0, 1)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'binary_performance_by_source.png', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'binary_performance_by_source.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: binary_performance_by_source.png/pdf")

def plot_continuous_performance_by_source(continuous_df):
    """Plot continuous regression performance by data source."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Aggregate by source
    source_agg = continuous_df.groupby('source').agg({
        'rmse': ['mean', 'std', 'count'],
        'r2': ['mean', 'std'],
        'pearson_r': ['mean', 'std']
    }).reset_index()
    source_agg.columns = ['source', 'rmse_mean', 'rmse_std', 'n_evals', 'r2_mean', 'r2_std', 'r_mean', 'r_std']
    source_agg = source_agg.sort_values('r2_mean', ascending=True)

    y_pos = range(len(source_agg))

    # RMSE plot (lower is better, so sort differently)
    ax = axes[0]
    rmse_sorted = source_agg.sort_values('rmse_mean', ascending=False)
    y_pos_rmse = range(len(rmse_sorted))
    ax.barh(y_pos_rmse, rmse_sorted['rmse_mean'], xerr=rmse_sorted['rmse_std'], capsize=3, color='indianred', alpha=0.8)
    ax.set_yticks(y_pos_rmse)
    ax.set_yticklabels(rmse_sorted['source'], fontsize=8)
    ax.set_xlabel('RMSE')
    ax.set_title('RMSE by Source (lower is better)')

    # R2 plot
    ax = axes[1]
    ax.barh(y_pos, source_agg['r2_mean'], xerr=source_agg['r2_std'], capsize=3, color='mediumpurple', alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(source_agg['source'], fontsize=8)
    ax.set_xlabel('R² Score')
    ax.set_title('R² by Source')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    # Pearson r plot
    ax = axes[2]
    r_data = source_agg.dropna(subset=['r_mean'])
    y_pos_r = range(len(r_data))
    ax.barh(y_pos_r, r_data['r_mean'], xerr=r_data['r_std'], capsize=3, color='teal', alpha=0.8)
    ax.set_yticks(y_pos_r)
    ax.set_yticklabels(r_data['source'], fontsize=8)
    ax.set_xlabel('Pearson r')
    ax.set_title('Correlation by Source')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'continuous_performance_by_source.png', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'continuous_performance_by_source.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: continuous_performance_by_source.png/pdf")

def plot_overall_distribution(binary_df, continuous_df):
    """Plot overall metric distributions."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))

    # Binary metrics
    if binary_df is not None and len(binary_df) > 0:
        axes[0, 0].hist(binary_df['accuracy'], bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        axes[0, 0].axvline(binary_df['accuracy'].mean(), color='red', linestyle='--', label=f'Mean: {binary_df["accuracy"].mean():.3f}')
        axes[0, 0].set_xlabel('Accuracy')
        axes[0, 0].set_ylabel('Count')
        axes[0, 0].set_title('Binary: Accuracy Distribution')
        axes[0, 0].legend()

        axes[0, 1].hist(binary_df['f1'], bins=20, color='darkorange', alpha=0.7, edgecolor='black')
        axes[0, 1].axvline(binary_df['f1'].mean(), color='red', linestyle='--', label=f'Mean: {binary_df["f1"].mean():.3f}')
        axes[0, 1].set_xlabel('F1 Score')
        axes[0, 1].set_title('Binary: F1 Distribution')
        axes[0, 1].legend()

        auc_vals = binary_df['auc'].dropna()
        if len(auc_vals) > 0:
            axes[0, 2].hist(auc_vals, bins=20, color='forestgreen', alpha=0.7, edgecolor='black')
            axes[0, 2].axvline(auc_vals.mean(), color='red', linestyle='--', label=f'Mean: {auc_vals.mean():.3f}')
            axes[0, 2].set_xlabel('AUC-ROC')
            axes[0, 2].set_title('Binary: AUC Distribution')
            axes[0, 2].legend()

    # Continuous metrics
    if continuous_df is not None and len(continuous_df) > 0:
        axes[1, 0].hist(continuous_df['rmse'], bins=20, color='indianred', alpha=0.7, edgecolor='black')
        axes[1, 0].axvline(continuous_df['rmse'].mean(), color='red', linestyle='--', label=f'Mean: {continuous_df["rmse"].mean():.3f}')
        axes[1, 0].set_xlabel('RMSE')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Continuous: RMSE Distribution')
        axes[1, 0].legend()

        axes[1, 1].hist(continuous_df['r2'], bins=20, color='mediumpurple', alpha=0.7, edgecolor='black')
        axes[1, 1].axvline(continuous_df['r2'].mean(), color='red', linestyle='--', label=f'Mean: {continuous_df["r2"].mean():.3f}')
        axes[1, 1].set_xlabel('R² Score')
        axes[1, 1].set_title('Continuous: R² Distribution')
        axes[1, 1].legend()

        r_vals = continuous_df['pearson_r'].dropna()
        if len(r_vals) > 0:
            axes[1, 2].hist(r_vals, bins=20, color='teal', alpha=0.7, edgecolor='black')
            axes[1, 2].axvline(r_vals.mean(), color='red', linestyle='--', label=f'Mean: {r_vals.mean():.3f}')
            axes[1, 2].set_xlabel('Pearson r')
            axes[1, 2].set_title('Continuous: Correlation Distribution')
            axes[1, 2].legend()

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'metric_distributions.png', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'metric_distributions.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: metric_distributions.png/pdf")

def plot_sample_size_vs_performance(binary_df, continuous_df):
    """Plot relationship between sample size and performance."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if binary_df is not None and len(binary_df) > 0:
        ax = axes[0]
        ax.scatter(binary_df['n_samples'], binary_df['accuracy'], alpha=0.6, c='steelblue', s=50)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('Accuracy')
        ax.set_title('Binary: Sample Size vs Accuracy')
        ax.set_xscale('log')

    if continuous_df is not None and len(continuous_df) > 0:
        ax = axes[1]
        ax.scatter(continuous_df['n_samples'], continuous_df['r2'], alpha=0.6, c='mediumpurple', s=50)
        ax.set_xlabel('Number of Samples')
        ax.set_ylabel('R² Score')
        ax.set_title('Continuous: Sample Size vs R²')
        ax.set_xscale('log')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'sample_size_vs_performance.png', bbox_inches='tight')
    plt.savefig(FIGURES_DIR / 'sample_size_vs_performance.pdf', bbox_inches='tight')
    plt.close()
    print("Saved: sample_size_vs_performance.png/pdf")

def generate_summary_table(results):
    """Generate a summary table for publication."""
    rows = []

    if 'binary' in results:
        binary_df = results['binary']
        for source in binary_df['source'].unique():
            source_data = binary_df[binary_df['source'] == source]
            auc_vals = source_data['auc'].dropna()
            rows.append({
                'Task Type': 'Binary Classification',
                'Source': source,
                'N Evaluations': len(source_data),
                'Total Samples': source_data['n_samples'].sum(),
                'Accuracy': f"{source_data['accuracy'].mean():.3f} ± {source_data['accuracy'].std():.3f}",
                'F1': f"{source_data['f1'].mean():.3f} ± {source_data['f1'].std():.3f}",
                'AUC': f"{auc_vals.mean():.3f} ± {auc_vals.std():.3f}" if len(auc_vals) > 0 else "N/A"
            })

    if 'continuous' in results:
        continuous_df = results['continuous']
        for source in continuous_df['source'].unique():
            source_data = continuous_df[continuous_df['source'] == source]
            r_vals = source_data['pearson_r'].dropna()
            rows.append({
                'Task Type': 'Continuous Regression',
                'Source': source,
                'N Evaluations': len(source_data),
                'Total Samples': source_data['n_samples'].sum(),
                'RMSE': f"{source_data['rmse'].mean():.3f} ± {source_data['rmse'].std():.3f}",
                'R²': f"{source_data['r2'].mean():.3f} ± {source_data['r2'].std():.3f}",
                'Pearson r': f"{r_vals.mean():.3f} ± {r_vals.std():.3f}" if len(r_vals) > 0 else "N/A"
            })

    summary_df = pd.DataFrame(rows)
    summary_df.to_csv(FIGURES_DIR / 'publication_summary_table.csv', index=False)
    print("Saved: publication_summary_table.csv")
    return summary_df

def main():
    """Generate all figures."""
    print("=" * 60)
    print("Generating ToxTransformer Benchmark Figures")
    print("=" * 60)

    results = load_results()

    if not results:
        print("No results found. Run evaluation first.")
        return

    binary_df = results.get('binary')
    continuous_df = results.get('continuous')

    if binary_df is not None and len(binary_df) > 0:
        plot_binary_performance_by_source(binary_df)

    if continuous_df is not None and len(continuous_df) > 0:
        plot_continuous_performance_by_source(continuous_df)

    if binary_df is not None or continuous_df is not None:
        plot_overall_distribution(binary_df, continuous_df)
        plot_sample_size_vs_performance(binary_df, continuous_df)

    summary_table = generate_summary_table(results)
    print("\nSummary Table:")
    print(summary_table.to_string())

    print(f"\nFigures saved to: {FIGURES_DIR}/")

if __name__ == "__main__":
    main()
