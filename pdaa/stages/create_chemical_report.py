import os
import itertools as it
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

cachedir = pathlib.Path('cache') / 'create_chemical_report'
cachedir.mkdir(parents=True, exist_ok=True)

chemical = 'dehp'
chemical_inchi = pubchem.lookup_chemical_inchi(chemical)
prompt = "reproductive toxicity"

# region build resources ===============================================================
# lookup all faissindex uris and get their embeddings
uri_faissindex = sparql.Query(pdaa.pdaa_graph) \
    .select_typed({'uri': str, 'index': int}) \
    .where('?faissindex <http://purl.org/dc/elements/1.1/has_identifier> ?uri') \
    .where('?faissindex a toxindex:faiss_index') \
    .where('?faissindex rdf:value ?index') \
    .execute()
uri_faissindex['embedding'] = uri_faissindex['index'].map(lambda i: pdaa.faiss_index.reconstruct(i))

# find the has_adverse_outcome ao that is in the same pathway as a given has_molecular_initiating_event mie
aop_mie_ao = sparql.Query(pdaa.aop_graph) \
    .select('aop', 'mie', 'ao') \
    .where('?aop aop:has_adverse_outcome ?ao') \
    .where('?aop aop:has_molecular_initiating_event ?mie') \
    .execute()

# find uris that has_identifier ?x where ?x is a toxindex:property_token
proptoken_uris = sparql.Query(pdaa.pdaa_graph) \
    .select_typed({'uri': str, 'proptoken': str, 'token': str}) \
    .where('?proptoken <http://purl.org/dc/elements/1.1/has_identifier> ?uri') \
    .where('?proptoken a <http://toxindex.com/ontology/predicted_property>') \
    .where('?proptoken rdf:value ?token') \
    .execute()

def get_prompt_similars(prompt, target_uris, top_k_to_search=1000):
    """Find most similar target URIs to a text prompt using FAISS embeddings.
    
    Args:
        prompt (str): Text prompt to compare against
        target_uris (list): List of URIs to search within
        top_k_to_search (int): Number of nearest neighbors to search before filtering to target URIs
        
    Returns:
        DataFrame with columns ['uri', 'similarity'] containing matches
    """
    # Get embeddings and search FAISS index
    prompt_embedding = np.array(openai_utils.embed(prompt))[np.newaxis, :]
    distances, indices = pdaa.faiss_index.search(prompt_embedding, top_k_to_search)
    
    # Filter to target URIs and format results
    target_indices = set(uri_faissindex[uri_faissindex['uri'].isin(target_uris)]['index'])
    matches = [(d, i) for d, i in zip(distances[0], indices[0]) if i in target_indices]
    
    if not matches:
        return pd.DataFrame(columns=['uri', 'similarity'])
        
    results = pd.DataFrame(matches, columns=['similarity', 'index'])
    return (results.merge(uri_faissindex, on='index')
            .groupby('uri')['similarity']
            .max()
            .reset_index()[['uri', 'similarity']])

def get_uri_similars(source_uris, target_uris):
    """Find most similar target URIs to source URIs using FAISS embeddings.
    
    Args:
        source_uris (list): List of source URIs to compare from
        target_uris (list): List of target URIs to search within
        top_k_to_search (int): Number of nearest neighbors to search before filtering
        
    Returns:
        DataFrame with columns ['source_uri', 'target_uri', 'similarity'] containing matches
    """
    # Get source embeddings from uri_faissindex
    source_faissindex = uri_faissindex[uri_faissindex['uri'].isin(source_uris)].reset_index()
    source_embeddings = np.vstack(source_faissindex['embedding'])
    
    # Get target embeddings and create temporary FAISS index
    target_faissindex = uri_faissindex[uri_faissindex['uri'].isin(target_uris)].reset_index()
    target_embeddings = np.vstack(target_faissindex['embedding'])
    target_faiss = faiss.IndexFlatIP(target_embeddings.shape[1])
    target_faiss.add(target_embeddings)
    
    # Search for nearest neighbors
    distances, indices = target_faiss.search(source_embeddings, target_embeddings.shape[0])
    
    # Convert to dataframe and expand pairs
    results = pd.DataFrame({
        'source_uri': np.repeat(source_faissindex['uri'].values, indices.shape[1]),
        'target_uri': target_faissindex.iloc[indices.ravel()]['uri'].values,
        'similarity': distances.ravel()
    })
    return results.groupby(['source_uri','target_uri'])['similarity'].max().reset_index()

def predict_predicted_property_uris(chemical_inchi, predicted_property_identifiers):
    tmp_proptoken = proptoken_uris[proptoken_uris['uri'].isin(predicted_property_identifiers)][['uri','token']]
    tmp_proptoken['int_token'] = tmp_proptoken['token'].astype(int)

    inchi_tok_pairs = set([(chemical_inchi, int(t)) for t in tmp_proptoken['int_token'].tolist()])
    predictions = pdaa.get_predictions_with_sqlite_cache(inchi_tok_pairs)
    prediction_df = pd.DataFrame(predictions, columns=['inchi', 'int_token', 'prediction'])

    result = prediction_df.merge(tmp_proptoken, on='int_token', how='inner')
    return result[['uri','prediction']]

adverse_outcomes = set(aop_mie_ao['ao'].tolist())

# get relevant adverse outcomes and their MIEs
prompt_adverse_outcomes = get_prompt_similars(prompt, adverse_outcomes, top_k_to_search=2000).query('similarity > 0.4')
prompt_ao_mie = aop_mie_ao[aop_mie_ao['ao'].isin(prompt_adverse_outcomes['uri'])][['ao','mie']].drop_duplicates()

# get relevant predicted properties for given MIEs
predicted_property_identifiers = proptoken_uris['uri'].unique()
relevant_mie = prompt_ao_mie['mie'].unique()
mie_proptoken_id_simtable = get_uri_similars(relevant_mie, predicted_property_identifiers).query('similarity > 0.4')
mie_proptoken_id_simtable.columns = ['mie','property_token_id','similarity']

# get the chemical predicted properties
relevant_property_tokens = mie_proptoken_id_simtable['property_token_id'].unique()
predictions_df = predict_predicted_property_uris(chemical_inchi, relevant_property_tokens)[['uri','prediction']]
predictions_df = predictions_df.query('prediction > 0.8')[['uri','prediction']]
predictions_df.columns = ['property_token_id','prediction']

# generate property_token_id weight for each mie
mie_proptoken_weight = mie_proptoken_id_simtable.merge(predictions_df, on='property_token_id', how='inner')
mie_proptoken_weight['weight'] = mie_proptoken_weight['similarity'] * mie_proptoken_weight['prediction']
mie_proptoken_weight.columns = ['mie', 'property_token_id', 'similarity', 'prediction', 'weight']

# get adverse outcome weights
ao_mie_weight = prompt_ao_mie.merge(mie_proptoken_weight, on='mie', how='inner')
ao_mie_weight = ao_mie_weight.groupby('ao')['weight'].sum().reset_index().sort_values('weight', ascending=False)

# uri titles
title_pred = "<http://purl.org/dc/elements/1.1/title>"
uri_titles = pdaa.pdaa_query(f"SELECT ?uri ?title WHERE {{ ?uri {title_pred} ?title . }}")

results = ao_mie_weight.merge(uri_titles, left_on='ao', right_on='uri', how='inner')
ao_results = results[['ao','title','weight']]

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