import os
import numpy as np
import requests
import sys
sys.path.append('./')
import stages.utils.openai as openai_utils
import stages.utils.chemprop as chemprop
import stages.utils.pdaa as pdaa
import stages.utils.pubchem as pubchem
import stages.utils.sparql as sparql

import biobricks as bb
import pathlib
import pandas as pd
import faiss
from rdflib import URIRef

cachedir = pathlib.Path('cache') / 'entity_similarity'
cachedir.mkdir(parents=True, exist_ok=True)

# region faiss cosine matrix ===============================================================
# lookup all faissindex uris and get their embeddings
# get faiss indexes of uris
uri_faissindex = sparql.Query(pdaa.pdaa_graph) \
    .select('uri', 'faissindex', 'index') \
    .where('?uri <http://edamontology.org/has_identifier> ?faissindex') \
    .where('?faissindex a <http://toxindex.com/ontology/faiss_index>') \
    .where('?faissindex <http://www.w3.org/2000/01/rdf-schema#label> ?index') \
    .execute()

# get aops [ uri, faissindex]
aops = pdaa.aopwiki_query("SELECT ?uri WHERE { ?uri a aop:AdverseOutcomePathway . }")
aop_faissindex = aops.merge(uri_faissindex, on='uri', how='inner')
aop_faissindex['index'] = aop_faissindex['index'].astype(int)
aop_faissindex['embedding'] = [pdaa.faiss_index.reconstruct(int(i)) for i in aop_faissindex['index'].values]

# list all the types in the aopwiki
aopwiki_types = pdaa.aopwiki_query("SELECT ?type WHERE { ?uri a ?type . }")
aopwiki_types = aopwiki_types.drop_duplicates()
aopwiki_types.to_csv(cachedir / 'aopwiki_types.csv', index=False)

aop_mie = pdaa.aopwiki_query("SELECT ?uri WHERE { ?sub aop:has_molecular_initiating_event ?uri . }")
aop_mie = aop_mie.drop_duplicates()
aop_mie_faissindex = aop_mie.merge(uri_faissindex, on='uri', how='inner')[['uri','index']]
aop_mie_faissindex['embedding'] = [pdaa.faiss_index.reconstruct(int(i)) for i in aop_mie_faissindex['index'].values]

key_events = pdaa.aopwiki_query("SELECT ?uri WHERE { ?uri a aop:KeyEvent . }")
key_events_predicates = pdaa.aopwiki_query("""
    SELECT ?uri ?predicate WHERE { 
        ?uri ?predicate <https://identifiers.org/aop.events/1617> . 
    }""")

# find the has_adverse_outcome ao that is in the same pathway as a given has_molecular_initiating_event mie
aop_mie_ao = pdaa.aopwiki_query("""
    SELECT ?aop ?mie ?ao WHERE { 
        ?aop aop:has_adverse_outcome ?ao . 
        ?aop aop:has_molecular_initiating_event ?mie . 
    }""")[['aop','mie','ao']]

# find uris that has_identifier ?x where ?x is a toxindex:property_token
proptoken_uris = pdaa.pdaa_query("""
    SELECT ?uri ?proptoken ?token WHERE { 
        ?uri EDAM:has_identifier ?proptoken . 
        ?proptoken a <http://toxindex.com/ontology/property> . 
        ?proptoken rdfs:label ?token .
    }""")
proptoken_faissindex = proptoken_uris.merge(uri_faissindex, on='uri', how='inner')[['uri','proptoken','index','token']]

proptoken_embeddings = np.vstack([pdaa.faiss_index.reconstruct(int(i)) for i in proptoken_faissindex['index'].values])
proptoken_faiss = faiss.IndexFlatIP(proptoken_embeddings.shape[1])
proptoken_faiss.add(proptoken_embeddings)

# Get similarities between property tokens and aop_ao_faissindex
D, I = proptoken_faiss.search(np.vstack(aop_mie_faissindex['embedding'].values), k=len(proptoken_faissindex))
D, I = D.tolist(), I.tolist()

# Create similarity matrix with property tokens and AOPs
simtable = []
for i in aop_mie_faissindex.index:
    aop_mie_uri = aop_mie_faissindex.iloc[i]['uri']
    for dist, prop_idx in zip(D[i], I[i]):
        if dist >= 0.4:  # Only keep similarities >= 40%
            prop_uri = proptoken_faissindex.iloc[prop_idx]['uri']
            prop_token = int(proptoken_faissindex.iloc[prop_idx]['token'])
            simtable.append({
                'mie': aop_mie_uri,
                'property_uri': prop_uri,
                'property_token': prop_token,
                'similarity': dist
            })

aop_mie_simtable = pd.DataFrame(simtable)
aop_mie_simtable = aop_mie_simtable.groupby(['mie','property_token'])['similarity'].max().reset_index()
aop_mie_simtable.to_csv(cachedir / 'aop_mie_simtable.csv', index=False)

# uri titles
aop_titles = pdaa.aopwiki_query("SELECT ?uri ?title WHERE { ?uri purl:title ?title . }")
prop_titles = pdaa.pdaa_query("SELECT ?uri ?title WHERE { ?uri purl:title ?title . }")
uri_titles = prop_titles.merge(aop_titles, on='uri', how='inner')

# endregion

# region AOPWIKI PREDICTIONS ===============================================================
# a function that takes a chemical name and returns of relevant adverse outcomes pathways
def get_aopwiki_predictions(chemical_name):
    predictions : list[tuple[str,int,float]] = get_all_property_predictions(inchi)
    pos_predictions = pd.DataFrame(predictions, columns=['inchi', 'property_token', 'prediction'])
    pos_predictions = pos_predictions[pos_predictions['prediction'] > 0.8]
    
    chem_aop_mie_simtable = aop_mie_simtable.merge(pos_predictions, on='property_token', how='inner')
    chem_aop_mie_simtable['weight'] = chem_aop_mie_simtable['similarity'] * chem_aop_mie_simtable['prediction']
    
    aop_mie_weight = chem_aop_mie_simtable.groupby('mie')['similarity'].sum().reset_index().sort_values('similarity', ascending=False)
    
    aop_ao_weight = aop_mie_simtable.merge(aop_mie_ao, on='mie', how='inner')[['mie','ao','similarity']]
    aop_ao_weight = aop_ao_weight.groupby('ao')['similarity'].mean().reset_index().sort_values('similarity', ascending=False)
    

    res = pdaa.aopwiki_query("SELECT ?p ?o WHERE { <https://identifiers.org/aop/43> ?p ?o }")[['o','p']]
    

# region AOP SPECIFIC PREDICTIONS ==========================================================
# a function that takes a chemical name and an AOP and returns relevant predictions
def get_aop_specific_predictions(chemical_name, aop):
    predictions = pdaa.get_predictions_with_sqlite_cache(inchi)

# region PRIORITY PHTALATE HEATMAP =======================================================
import rdkit, rdkit.Chem, rdkit.Chem.AllChem, rdkit.DataStructs, rdkit.Chem.rdFingerprintGenerator
import itertools as it
dehp = rdkit.Chem.MolFromSmiles('CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC')
diup = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCCC(C)C')
dtdp = rdkit.Chem.MolFromSmiles('CCCCCCCCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCCCCCCC')
didp = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCC(C)C')
dinp = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCC(C)C')

phthalate_inchi_names = ['dehp', 'diup', 'dtdp', 'didp', 'dinp']
priority_phthalates = [dehp, diup, dtdp, didp, dinp]
phthalates_inchi = [rdkit.Chem.inchi.MolToInchi(p) for p in priority_phthalates]

# get all predictions 
inchi_tok_pairs = list(it.product(phthalates_inchi, properties_df['property_token'].astype(int).values))
inchi_tok_pairs = [(i, int(t)) for i,t in inchi_tok_pairs]
predictions : list[tuple[str,int,float]] = pdaa.get_predictions_with_sqlite_cache(inchi_tok_pairs)

# create a heatmap with phthalates on the x axis and properties on the y axis
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import math


# Convert predictions list to matrix form
pred_matrix = pd.DataFrame(predictions, columns=['inchi', 'property_token', 'value'])
pred_matrix = pred_matrix.merge(propcat, on='property_token', how='inner')

# create one heatmap
# Pivot the data to create a matrix for each category
categories = list(['neurotoxicity','carcinogenicity','chronic toxicity','mutagenicity','endocrine disruption','hepatotoxicity','reproductive toxicity',
                  'developmental toxicity','genotoxicity','eye irritation','immunotoxicity','aquatic toxicity',
                  'environmental toxicity','sub-chronic toxicity','acute oral toxicity',
                  'nephrotoxicity','ecotoxicity','acute inhalation toxicity','skin irritation','dermal absorption'])
pred_matrix = pred_matrix[pred_matrix['category'].isin(categories)]


# there should be at most 3 columns
# Create individual clustermaps for each category and save them
clustermap_figs = []
image_names = []
for cat in categories:
    cat_data = pred_matrix[pred_matrix['category'] == cat].pivot(
        index='property_token', 
        columns='inchi', 
        values='value'
    )
    
    # Skip if no data for this category
    if cat_data.empty:
        continue
        
    # Create clustermap
    g = sns.clustermap(
        cat_data,
        cmap='viridis',
        dendrogram_ratio=(.2, .1),
        figsize=(6, 4),
        yticklabels=False,
        xticklabels=False,
        cbar=False
    )
    
    # Add title
    g.ax_heatmap.set_title(cat)
    
    # Save figure
    path = f'heatmap_{cat.lower().replace(" ", "_")}.png'
    image_names.append(path)
    plt.savefig(path)
    plt.close()

images = []
for cat in categories:
    try:
        img_path = f'heatmap_{cat.lower().replace(" ", "_")}.png'
        images.append(Image.open(img_path))
    except:
        continue

# Calculate grid dimensions
n_images = len(images)
n_cols = 3
n_rows = math.ceil(n_images / n_cols)

# Create blank canvas
cell_width = images[0].width
cell_height = images[0].height
canvas = Image.new('RGB', (cell_width * n_cols, cell_height * n_rows))

# Paste images into grid
for idx, img in enumerate(images):
    row = idx // n_cols
    col = idx % n_cols
    canvas.paste(img, (col * cell_width, row * cell_height))

# Save final composite
canvas.save('category_heatmaps.png')

# remove the temporary files
for img in image_names:
    os.remove(img)

# CREATE GLOBAL HEATMAP ===============================================================
# For the main clustered heatmap, pivot the full matrix
# Create a mapping of InChI to phthalate names
phthalate_names = {
    rdkit.Chem.inchi.MolToInchi(dehp): 'DEHP',
    rdkit.Chem.inchi.MolToInchi(diup): 'DIUP', 
    rdkit.Chem.inchi.MolToInchi(dtdp): 'DTDP',
    rdkit.Chem.inchi.MolToInchi(didp): 'DIDP',
    rdkit.Chem.inchi.MolToInchi(dinp): 'DINP'
}

piv_pred_matrix = pred_matrix[['property_token','inchi','value']].drop_duplicates().pivot(
    index='property_token',
    columns='inchi', 
    values='value'
)

# Rename the columns using the phthalate names
piv_pred_matrix = piv_pred_matrix.rename(columns=phthalate_names)

g = sns.clustermap(
    piv_pred_matrix,
    cmap='viridis', 
    dendrogram_ratio=(.2, .1),
    cbar_pos=(0.02, .32, .03, .2),
    figsize=(6, 12),  # Increased height from 4 to 12
    yticklabels=False,
    xticklabels=True  # Show the phthalate name labels
)

plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right')
plt.savefig('test.png', bbox_inches='tight')

# region MORE INCHI HEATMAP ===============================================================
import rdkit, rdkit.Chem, rdkit.Chem.AllChem, rdkit.DataStructs, rdkit.Chem.rdFingerprintGenerator
import itertools as it
dehp = rdkit.Chem.MolFromSmiles('CCCCC(CC)COC(=O)C1=CC=CC=C1C(=O)OCC(CC)CCCC')
diup = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCCC(C)C')
dtdp = rdkit.Chem.MolFromSmiles('CCCCCCCCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCCCCCCC')
didp = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCCC(C)C')
dinp = rdkit.Chem.MolFromSmiles('CC(C)CCCCCCOC(=O)C1=CC=CC=C1C(=O)OCCCCCCC(C)C')

raw_df = pd.read_parquet('cache/priority_phthalates/priority_phthalates.parquet')
top_df = raw_df.sort_values(by='max_similarity', ascending=False)[['inchi', 'max_similarity']].drop_duplicates()
inchi_list = top_df['inchi'][:100].unique().tolist()

priority_phthalates = [dehp, diup, dtdp, didp, dinp]
phthalates_inchi = [rdkit.Chem.inchi.MolToInchi(p) for p in priority_phthalates]

# get all predictions 
inchi_tok_pairs = list(it.product(phthalates_inchi, properties_df['property_token'].astype(int).values))
inchi_tok_pairs = [(i, int(t)) for i,t in inchi_tok_pairs]
predictions : list[tuple[str,int,float]] = pdaa.get_predictions_with_sqlite_cache(inchi_tok_pairs)

# create a heatmap with phthalates on the x axis and properties on the y axis
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import math


# Convert predictions list to matrix form
pred_matrix = pd.DataFrame(predictions, columns=['inchi', 'property_token', 'value'])
pred_matrix = pred_matrix.merge(propcat, on='property_token', how='inner')

# create one heatmap
# Pivot the data to create a matrix for each category
categories = list(['neurotoxicity','carcinogenicity','chronic toxicity','mutagenicity','endocrine disruption','hepatotoxicity','reproductive toxicity',
                  'developmental toxicity','genotoxicity','eye irritation','immunotoxicity','aquatic toxicity',
                  'environmental toxicity','sub-chronic toxicity','acute oral toxicity',
                  'nephrotoxicity','ecotoxicity','acute inhalation toxicity','skin irritation','dermal absorption'])
pred_matrix = pred_matrix[pred_matrix['category'].isin(categories)]


# there should be at most 3 columns
# Create individual clustermaps for each category and save them
clustermap_figs = []
image_names = []
for cat in categories:
    cat_data = pred_matrix[pred_matrix['category'] == cat].pivot(
        index='property_token', 
        columns='inchi', 
        values='value'
    )
    
    # Skip if no data for this category
    if cat_data.empty:
        continue
        
    # Create clustermap
    g = sns.clustermap(
        cat_data,
        cmap='viridis',
        dendrogram_ratio=(.2, .1),
        figsize=(6, 4),
        yticklabels=False,
        xticklabels=False,
        cbar=False
    )
    
    # Add title
    g.ax_heatmap.set_title(cat)
    
    # Save figure
    path = f'heatmap_{cat.lower().replace(" ", "_")}.png'
    image_names.append(path)
    plt.savefig(path)
    plt.close()

images = []
for cat in categories:
    try:
        img_path = f'heatmap_{cat.lower().replace(" ", "_")}.png'
        images.append(Image.open(img_path))
    except:
        continue

# Calculate grid dimensions
n_images = len(images)
n_cols = 3
n_rows = math.ceil(n_images / n_cols)

# Create blank canvas
cell_width = images[0].width
cell_height = images[0].height
canvas = Image.new('RGB', (cell_width * n_cols, cell_height * n_rows))

# Paste images into grid
for idx, img in enumerate(images):
    row = idx // n_cols
    col = idx % n_cols
    canvas.paste(img, (col * cell_width, row * cell_height))

# Save final composite
canvas.save('category_heatmaps.png')

# remove the temporary files
for img in image_names:
    os.remove(img)

# For the main clustered heatmap, pivot the full matrix
piv_pred_matrix = pred_matrix[['property_token','inchi','value']].drop_duplicates().pivot(
    index='property_token',
    columns='inchi',
    values='value'
)

g = sns.clustermap(
    piv_pred_matrix, # Transpose the matrix to flip axes
    cmap='viridis',
    dendrogram_ratio=(.2, .1),
    cbar_pos=(0.02, .32, .03, .2),
    figsize=(12, 4), # Adjust figure size to make squares more square
    yticklabels=True, # Keep phthalate labels
    xticklabels=False # Remove property token labels
)

plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0)
plt.savefig('test.png')