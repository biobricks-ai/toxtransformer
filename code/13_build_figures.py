import matplotlib.pyplot as plt, pandas as pd, seaborn as sns
import random

# SETUP =================================================================================
df = pd.read_parquet('cache/eval_multi_properties/multitask_metrics.parquet')
df.aggregate({'AUC': 'median', 'ACC': 'median', 'BAC': 'median', "cross_entropy_loss": 'median'})
df.groupby('nprops').aggregate({'AUC': 'median', 'ACC': 'median', 'BAC': 'median', "cross_entropy_loss": 'median', 'assay': 'nunique'})
df[df['NUM_POS'] > 100].groupby('nprops').aggregate({'AUC': 'median', 'ACC': 'median', 'BAC': 'median', "cross_entropy_loss": 'median', 'assay': 'nunique'})

# how many assays?
df['assay'].nunique()
auc = df['AUC'].median()

# AUC HISTOGRAM =========================================================================
def auc_histogram(df,nprops):
    assays = df.groupby('assay').filter(lambda x: x['nprops'].nunique() == nprops)['assay'].unique()
    plotdf = df[df['nprops'].isin(list(range(nprops))) & df['assay'].isin(assays)]

    plt.style.use('dark_background')
    g = sns.FacetGrid(plotdf, col='nprops', height=5, aspect=1.3, col_wrap=3, sharex=False, sharey=False)
    g.map(plt.hist, 'AUC', bins=30, alpha=1.0, edgecolor='white', linewidth=1, color='#16A085')

    for ax, nprop in zip(g.axes.flatten(), [0, 1, 2, 3, 4]):
        median_auc = plotdf[plotdf['nprops'] == nprop]['AUC'].median()
        ax.axvline(median_auc, color='red', linestyle='dashed', linewidth=3)
        ax.annotate(f'Median AUC: {median_auc:.3f}', xy=(median_auc-0.125, 0.95), xycoords='axes fraction',
                    fontsize=12, color='red', fontweight='bold', ha='right', va='top')
        ax.set_title(f'Prior Properties = {nprop}', color='white', fontsize=18)
        ax.set_xlabel('AUC', color='white', fontsize=14)
        ax.set_ylabel('Number of Properties', color='white', fontsize=14)
        ax.tick_params(colors='white', labelsize=12)

    g.figure.suptitle('Histogram of AUC per Property', color='white', fontsize=22, fontweight='bold', y=0.98)
    plt.subplots_adjust(top=0.9)  # Adjust this value as needed to make room for the title
    g.figure.tight_layout(rect=[0, 0.03, 1, 0.9])  # Adjust the rect to ensure title is visible
    g.add_legend()
    plt.savefig('notebook/plots/multitask_transformer_metrics.png', facecolor='none', transparent=True)

auc_histogram(df, nprops=5)
