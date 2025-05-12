import pandas as pd, sqlite3, seaborn as sns, matplotlib.pyplot as plt, pathlib
import cvae.tokenizer
from matplotlib.colors import ListedColormap

#%% SETUP =================================================================================
outdir = pathlib.Path("cache/benchmarks")
outdir.mkdir(exist_ok=True, parents=True)

tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer/')

#%% SOURCE ASSERTIONS =======================================================================
# conn = sqlite3.connect('brick/cvae.sqlite')

# # # # check that every source in conn is also in evaldf
# prop_src = pd.read_sql("SELECT property_token,title,source FROM property p INNER JOIN source s on p.source_id = s.source_id", conn)
# prop_src['source'].value_counts()

# evaldf = pd.read_parquet('cache/eval_multi_properties/multitask_metrics.parquet')
# evaldf = evaldf.rename(columns={'assay': 'property_token'})

# # are all evaldf 'assay' in prop_src?
# assert evaldf['property_token'].isin(prop_src['property_token']).all()

# # get min and max of each tables 'property_token'
# print(evaldf['property_token'].min(), evaldf['property_token'].max())
# print(prop_src['property_token'].min(), prop_src['property_token'].max())

# tokassays = list(tokenizer.assay_indexes().values())
# print(f"Tokenizer assay indexes min: {min(tokassays)}, max: {max(tokassays)}")

# evaldf['property_token'].nunique()
# assert prop_src['property_token'].isin(evaldf['property_token']).all()
# badprops2 = prop_src[~prop_src['property_token'].isin(evaldf['property_token'])].reset_index(drop=True)
# badprops2.shape
# badprops2['source'].value_counts()

# goodprops = prop_src[prop_src['property_token'].isin(evaldf['property_token'])].reset_index(drop=True)
# goodprops['source'].value_counts()

#%% TOXCAST BENCHMARK ===========================================================
conn = sqlite3.connect('brick/cvae.sqlite')

# get all the property_tokens for tox21 properties
prop_src = pd.read_sql("SELECT property_token,title,source FROM property p INNER JOIN source s on p.source_id = s.source_id", conn)
# prop_src = prop_src[prop_src['source'] != 'Tox21'] #tox21 is redundant

# assert that each property_token has only one title
assert prop_src.groupby('property_token').size().max() == 1

# pull in multitask_metrics
evaldf = pd.read_parquet('cache/eval_multi_properties/multitask_metrics.parquet')
evaldf = evaldf.rename(columns={'assay': 'property_token'})
evaldf = evaldf.merge(prop_src, left_on='property_token', right_on='property_token', how='inner')

# get the median AUC for each property
evaldf.aggregate({'AUC': 'median','cross_entropy_loss':'median','property_token':'count'}) # 89% median auc, .482 median cross entropy loss
res = evaldf.groupby(['source','nprops']).aggregate({'AUC': 'median','property_token':'count'}).sort_values(by='AUC',ascending=False)
tox21 = evaldf[evaldf['source'] == 'Tox21'].groupby(['nprops'])\
    .aggregate({'AUC': 'median','property_token':'count'})\
    .sort_values(by='AUC',ascending=False)\
    .reset_index()
tox21.to_csv(outdir / 'tox21_auc_by_nprops.csv', index=False)

#              AUC  assay
# nprops                 
# 4       0.900132    110
# 3       0.902808    110
# 2       0.909173    110
# 1       0.871136     66
# 0       0.829932     18

# region SOURCE EVALUATIONS ================================
source_eval = evaldf.groupby('property_token').filter(lambda x: len(x['nprops'].unique()) == 5)
res = source_eval.groupby(['source','nprops']).aggregate({'AUC': 'median','property_token':'count'}).reset_index()

# add 'meanauc' column
res['meanauc'] = res.groupby('source')['AUC'].transform('median')

# pivot wider, make nprops into a column 
# pivot wider, make nprops into columns
pivot_res = res.pivot(index='source', columns='nprops', values='AUC')
pivot_res['mean_auc'] = pivot_res.mean(axis=1)
pivot_res = pivot_res.sort_values('mean_auc', ascending=False)
pivot_res = pivot_res.drop('mean_auc', axis=1)

# Create heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(pivot_res, annot=True, cmap='RdYlGn', center=0.75, vmin=0.5, vmax=1.0, fmt='.3f')
plt.title('AUC by Source and Number of Prior Properties')
plt.xlabel('Number of Prior Properties')
plt.ylabel('Source')
plt.tight_layout()
plt.savefig(outdir / 'source_nprops_heatmap.png')
plt.close()

# endregion

#%% CATEGORICAL BENCHMARK ======================================================

pcat = pd.read_sql("""SELECT s.source, property_token,title,category,strength FROM property p 
                       INNER JOIN property_category pc ON p.property_id = pc.property_id
                       INNER JOIN category c on pc.category_id = c.category_id
                       INNER JOIN source s on p.source_id = s.source_id """, 
                       conn)
    
evalcat = evaldf.merge(pcat, left_on=['property_token', 'source', 'title'], right_on=['property_token', 'source', 'title'], how='inner')\
    .query('strength > 8.0')

# query the nephrotoxicity category and read titles and AUCs
res = evalcat.query('category == "nephrotoxicity"')[['property_token','source','title','AUC','NUM_POS','NUM_NEG','strength']].sort_values(by='AUC',ascending=False)

# group by property_token and take the first row for each property
res = res.sort_values(by='AUC',ascending=False)

res = res.groupby('property_token').first().reset_index()

# iterate over res and print title and AUC, with 2 decimal precision
for i, row in res.iterrows():
    print(f'{row["strength"]}\t{row["AUC"]:.2f}\t{row["NUM_POS"]+row["NUM_NEG"]}\t{row["source"]}\t{row["title"]}')

# get the median AUC for each property
evalcat.aggregate({'AUC': 'median','property_token':'count'})
pdf = evalcat.groupby(['category','nprops']).aggregate({'AUC': 'median','property_token':'count'}).reset_index().sort_values(by=['category','AUC'],ascending=False)

# create a 'category_order' column with a numeric sorting categories by median AUC
pdf['category_order'] = pdf.groupby('category')['AUC'].transform('median')
pdf['num_nprops'] = pdf.groupby('category')['nprops'].transform('count')
pdf = pdf[pdf['num_nprops'] > 4]

# create heatmap of AUC's with range 0.5 to 1.0 red to green 
# rows should be categories, columns should be nprops
heatmap_data = pdf.pivot(index="category", columns="nprops", values="AUC")
heatmap_data['order'] = pdf.groupby('category')['category_order'].first()
heatmap_data.index = heatmap_data.index.map(lambda x: x[:25] if len(x) > 25 else x)
heatmap_data = heatmap_data.sort_values(by='order', ascending=False).drop(columns='order')

# Create the heatmap using seaborn
# cmap = sns(["#2a2a2a", "#486730", "#34eb37"], as_cmap=True)
# cmap = sns.diverging_palette(240, 120, s=75, l=40, n=3, center="dark", as_cmap=True)
cmap = ListedColormap(['#2a2a2a', '#486730', '#34eb37', '#d4af37'])

heatmap_data_colored = heatmap_data.map(lambda x: 0 if x <= 0.6 else (1 if x <= 0.7 else 2))

# Create the heatmap using seaborn
plt.figure(figsize=(12, 9))
ax = sns.heatmap(heatmap_data, cmap=cmap, annot=False, square=True, fmt=".2f",
                 linewidths=0.5, linecolor='black', cbar_kws={"shrink": .8})

plt.title('Heatmap of Median AUC by Category and Number of Properties', color='white', fontsize=20)
plt.xlabel('Number of Properties', fontsize=15, color='white')
plt.ylabel('Category', fontsize=15, color='white')

# Setting the tick labels color for row and column names
ax.tick_params(axis='x', colors='white')
ax.tick_params(axis='y', colors='white')

# Improve the appearance of the colorbar
cbar = ax.collections[0].colorbar
cbar.set_label('Median AUC', color='white')
cbar.ax.yaxis.set_tick_params(color='white')
cbar.ax.yaxis.set_ticklabels([f'{x:.2f}' for x in cbar.get_ticks()], color='white')

# Save the figure to a file
plt.savefig(outdir / 'category_auc_heatmap.png',facecolor='black',bbox_inches='tight',dpi=300)