import sys, os, pandas as pd, tqdm, sqlite3, seaborn as sns, matplotlib.pyplot as plt, sklearn.metrics
sys.path.append("./")
from flask_cvae.predictor import Predictor
from tqdm import tqdm
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, leaves_list
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm
import matplotlib.gridspec as gridspec

tqdm.pandas()

predictor : Predictor = Predictor('flask_cvae/predictions.sqlite')
def build_propdf():
    conn = sqlite3.connect('brick/cvae.sqlite')
    proptitle = pd.read_sql("SELECT property_token,title FROM property p", conn).groupby('property_token').first().reset_index()
    propdf = predictor._get_all_properties().groupby('property_token').first().reset_index()\
        .merge(proptitle, on='property_token', how='inner')

    # remove poor performing properties
    evaldf = pd.read_parquet('data/metrics/multitask_metrics.parquet')\
        .rename(columns={'assay':'property_token'})\
        .groupby('property_token').aggregate({'AUC':'median'})\
        .query('AUC > .7')\
        .reset_index()

    # remove irrelevant categories
    # return propdf.merge(evaldf, on='property_token', how='inner')\
    return propdf\
        .query('category != "kinetics (pharmacokinetics, toxicokinetics, adme, cmax, auc, etc)"')\
        .query('category != "chemical physical property"')\
        .rename(columns={'strength':'category_strength', 'title':'property_title','value':'known_value'})

propdf = build_propdf()

#%% SETUP =======================================================================
dehp = "InChI=1S/C24H38O4/c1-5-9-13-19(7-3)17-27-23(25)21-15-11-12-16-22(21)24(26)28-18-20(8-4)14-10-6-2/h11-12,15-16,19-20H,5-10,13-14,17-18H2,1-4H3"
alternatives = {
    'Di(2-ethylhexyl) phthalate': dehp,
    'Diisononyl phthalate':                     "InChI=1S/C26H42O4/c1-21(2)15-9-5-7-13-19-29-25(27)23-17-11-12-18-24(23)26(28)30-20-14-8-6-10-16-22(3)4/h11-12,17-18,21-22H,5-10,13-16,19-20H2,1-4H3",
    'Diisodecyl phthalate':                     "InChI=1S/C28H46O4/c1-23(2)17-11-7-5-9-15-21-31-27(29)25-19-13-14-20-26(25)28(30)32-22-16-10-6-8-12-18-24(3)4/h13-14,19-20,23-24H,5-12,15-18,21-22H2,1-4H3",
    'Dioctyl terephthalate':                    "InChI=1S/C24H38O4/c1-3-5-7-9-11-13-19-27-23(25)21-15-17-22(18-16-21)24(26)28-20-14-12-10-8-6-4-2/h15-18H,3-14,19-20H2,1-2H3",
    'Acetyl tributyl citrate':                  "InChI=1S/C20H34O8/c1-5-8-11-25-17(22)14-20(28-16(4)21,19(24)27-13-10-7-3)15-18(23)26-12-9-6-2/h5-15H2,1-4H3",
    'Trioctyl trimellitate':                    "InChI=1S/C33H54O6/c1-7-13-16-25(10-4)22-37-31(34)28-19-20-29(32(35)38-23-26(11-5)17-14-8-2)30(21-28)33(36)39-24-27(12-6)18-15-9-3/h19-21,25-27H,7-18,22-24H2,1-6H3",
    'Diisononyl cyclohexane-1,2-dicarboxylate': "InChI=1S/C26H48O4/c1-21(2)15-9-5-7-13-19-29-25(27)23-17-11-12-18-24(23)26(28)30-20-14-8-6-10-16-22(3)4/h21-24H,5-20H2,1-4H3",
    'Butyryl trihexyl citrate':                 "InChI=1S/C28H50O8/c1-5-9-12-15-19-33-25(30)22-28(36-24(29)18-8-4,27(32)35-21-17-14-11-7-3)23-26(31)34-20-16-13-10-6-2/h5-23H2,1-4H3",
    'Di(2-ethylhexyl) terephthalate':           "InChI=1S/C24H38O4/c1-5-9-11-19(7-3)17-27-23(25)21-13-15-22(16-14-21)24(26)28-18-20(8-4)12-10-6-2/h13-16,19-20H,5-12,17-18H2,1-4H3",
}

altdf = pd.DataFrame(alternatives.items(), columns=['name','inchi'])
altdf = altdf.assign(key=1).merge(propdf.assign(key=1), on='key').drop('key', axis=1)
# TODO This is just for testing
# altdf = altdf.head(5)
altdf['prediction'] = altdf.progress_apply(lambda x: predictor.cached_predict_property(x['inchi'], x['property_token']), axis=1)

known_dfs = []
for inchi in altdf['inchi'].unique():
    df = pd.DataFrame(predictor._get_known_properties(inchi))[['inchi','property_token','value']]
    known_dfs.append(df)
knowndf = pd.concat(known_dfs)

# count inchi, property_token in altdf
altdf = altdf.merge(knowndf, on=['inchi','property_token'], how='left')
altdf['value'] = altdf['value'].fillna('unknown')

# get sklearn bac and acc 
def compute_metrics(g):
    nprops = g['property_token'].nunique()
    numtrue = g['value'].sum()
    numfalse = len(g) - numtrue
    acc = sklearn.metrics.accuracy_score(g['value'], g['prediction'] > 0.5)
    return pd.Series({'accuracy': acc, 'numprops': nprops, 'numtrue': numtrue, 'numfalse': numfalse, 'numeval': len(g)})

# Group by 'inchi' and apply the metrics function
accdf = altdf.query('value in ["positive","negative"]').assign(value = altdf['value'] == 'positive')
accdf = accdf.query('category_strength > 8.0')
resdf = accdf.groupby(['category']).apply(compute_metrics)\
    .reset_index().sort_values('accuracy', ascending=False)\
    .query('numeval > 20')

resdf

#%% RANK ALTERNATIVES BY CATEGORICAL HAZARD ==========================================================
# can the models identify these alternatives as the least toxic?
# GPT4o says ATBC is least hazardous DEHP alternative

relevant_hazards = ['reproductive toxicity','endocrine disruption','mutagenicity','developmental toxicity','genotoxicity']
adf = altdf\
    .assign(likely_hazard = altdf['prediction'] > 0.1)\
    .query(f'category in {relevant_hazards}')\
    .query(f'category_strength > 8.0')\
    .rename(columns={'title':'property_title','value':'known_value'})\
    [['name','property_title','category','known_value','prediction','likely_hazard']]\
    .drop_duplicates()

# 153 STRONGLY CATEGORIZED PROPERTIES
adf['property_title'].nunique() 

# OVERALL ACTIVITY
# ATBC is our overall least 'active' by a wide margin
# DEHP is 4th, and significantly worse than ATBC
# Dioctyl terephthalate is our overall most 'active'
count_hazards = lambda x: pd.Series({'likely_hazards': x['likely_hazard'].sum()})
adf.groupby('name').apply(count_hazards).sort_values('likely_hazards').reset_index()

# ENDOCRINE DISRUPTION
# ATBC least active 4th 
# Dioctyl terephthalate worst in endocrine as well
adf.query('category == "endocrine disruption"')\
    .groupby(['name']).apply(count_hazards).sort_values('likely_hazards').reset_index()
    
#%% FIND PREVALENT PROPERTIES ==========================================================
predsum = lambda x: pd.Series({'prediction_sum': x['prediction'].sum() / len(alternatives)})
pprops = altdf[['source','property_title','data','category','category_strength','name','prediction']]\
    .query('category_strength > 8.0')\
    .drop_duplicates()\
    .groupby(['source','property_title','category']).apply(predsum)\
    .sort_values('prediction_sum',ascending=False)\
    .reset_index()

for category in pprops['category'].unique():    
    print()
    print(category)
    subprops = pprops.query(f'category == "{category}"').head(min(10, len(pprops)))
    for source, title, category, val in subprops.values:
        print(f"{val:.2f}\t{source}\t{title}")

#%% HEATMAP

altdf = pd.read_csv('altdf.csv', index_col=0)

pdf = altdf.merge(knowndf, on=['inchi','property_token'], how='left').drop_duplicates()
pdf['known_value'] = pdf['value'].fillna('unknown')
pdf['known_value'] = pdf['known_value'].map({'positive': 1, 'negative': -1, 'unknown': 0})

pdf = pdf.query('category=="endocrine disruption"').query('category_strength > 9')[['property_title','name','prediction','known_value']].drop_duplicates()
pdf['property_title'] = pdf['property_title'].map(lambda x: x[:50] if len(x) > 50 else x)

heatmap_known = pdf.pivot_table(values='known_value', columns='name', index='property_title', fill_value=0) 
heatmap_data = pdf.pivot_table(values='prediction', columns='name', index='property_title', fill_value=0)

# Compute the cosine distance and perform hierarchical clustering
row_dist = pdist(heatmap_data, metric='cosine')
col_dist = pdist(heatmap_data.T, metric='cosine')

row_linkage = linkage(row_dist, method='average')
col_linkage = linkage(col_dist, method='average')

row_order = leaves_list(row_linkage)
col_order = leaves_list(col_linkage)

# Reorder the data
heatmap_known = heatmap_known.iloc[row_order, col_order]
heatmap_data = heatmap_data.iloc[row_order, col_order]

def build_plot(heatmap_known, heatmap_data):
    # Create a custom colormap that handles -1 as a special color and ranges from blue (0) to red (1)
    special_colors = ["#FFFFFF"]  # Black for -1
    gradient_cmap = LinearSegmentedColormap.from_list("grad_cmap", ["#3182bd", "#de2d26"])  # Gradient from blue to red

    # Combine the special color and gradient colormap
    combined_colors = special_colors + [gradient_cmap(i / gradient_cmap.N) for i in range(gradient_cmap.N)]
    combined_cmap = ListedColormap(combined_colors)
    norm = BoundaryNorm([-1, 0, 1], len(combined_colors))

    # Create a figure with two axes, setting size in pixels
    fig, (ax0, ax1) = plt.subplots(ncols=2, figsize=(4, 12), gridspec_kw={'wspace': 0.25})  # Reduce space between plots

    # Plot heatmap for heatmap_known, make a cmap with -1 as white, 0 as "#3182bd" and red as "#de2d26"
    known_cmap = ListedColormap(["#FFFFFF", "#3182bd", "#de2d26"])
    sns.heatmap(heatmap_known, cmap=known_cmap, annot=False, square=True, fmt=".2f",
                linewidths=0.5, linecolor='black', cbar=False, ax=ax0)
    ax0.set_title('Known Data', color='white')
    ax0.tick_params(axis='x', colors='white')
    ax0.tick_params(axis='y', colors='white')

    # Plot heatmap for heatmap_data
    sns.heatmap(heatmap_data, cmap=gradient_cmap, annot=False, square=True, fmt=".2f",
                linewidths=0.5, linecolor='black', cbar=False, ax=ax1)
    ax1.set_title('Predicted Data', color='white')
    ax1.tick_params(axis='x', colors='white')
    ax1.tick_params(axis='y', left=False, labelleft=False)  # Remove y-axis labels
    # remove y-axis title
    ax1.set_ylabel('')

    # Improve the appearance of the colorbar
    # cbar = ax1.collections[0].colorbar
    # cbar.set_label('Value', color='white')
    # cbar.ax.yaxis.set_tick_params(color='white')
    # cbar.ax.yaxis.set_ticklabels([f'{x:.2f}' for x in cbar.get_ticks()], color='white')

    # Set background color to black
    fig.patch.set_facecolor('black')

    # Save the figure to a file
    os.makedirs('notebook/plots', exist_ok=True)
    plt.savefig('notebook/plots/test.png', facecolor='black', bbox_inches='tight', dpi=300)


build_plot(heatmap_known, heatmap_data)