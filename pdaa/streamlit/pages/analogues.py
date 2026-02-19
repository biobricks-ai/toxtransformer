import sys
sys.path.append('./')

import json
import streamlit as st
import pandas as pd
import pathlib
from scipy.cluster.hierarchy import linkage, leaves_list
import plotly.graph_objects as go
import stages.utils.pdaa as pdaa
import stages.utils.pubchem as pubchem
import stages.utils.sparql as sparql
import stages.utils.simple_cache as simple_cache
import rdflib
from concurrent.futures import ThreadPoolExecutor
from rdkit import Chem
from rdkit.Chem import Draw
import io

# Setup
cachedir = pathlib.Path("cache") / "notebooks" / "pages" / "analogues"
cachedir.mkdir(parents=True, exist_ok=True)
pubchemtools = pubchem.PubchemTools()

# Get property titles for dropdown
@simple_cache.simple_cache_df(cachedir / "get_uri_titles")
def get_uri_titles():
    uri_titles = (
        sparql.Query(pdaa.pdaa_graph, cachedir / "pdaa_graph")
        .select_typed({"uri": str, "title": str})
        .where(f"?ppuri <http://purl.org/dc/elements/1.1/title> ?title")
        .where(f"?ppuri <{rdflib.RDF.type}> toxindex:predicted_property")
        .where(f"?ppuri <http://purl.org/dc/elements/1.1/has_identifier> ?uri")
        .cache_execute()
        .groupby("uri")
        .first()
        .reset_index()
    )
    # Ensure unique titles
    for title in uri_titles["title"].value_counts()[uri_titles["title"].value_counts() > 1].index:
        mask = uri_titles["title"] == title
        uri_titles.loc[mask, "title"] = [f"{title}_{i+1}" for i in range(sum(mask))]
    return uri_titles

def draw_molecule(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    img = Draw.MolToImage(mol)
    return img

def get_predictions_for_chemical(inchi):
    predictions = []
    with ThreadPoolExecutor() as executor:
        predictions.extend(executor.submit(pdaa.predict_all_properties_with_sqlite_cache, [inchi]).result())
    
    preds_df = pd.DataFrame(predictions, columns=['inchi', 'token', 'prediction'])
    preds_df = preds_df.merge(pdaa.proptoken_uris[['uri','token']], on='token')[['uri','inchi','prediction']]
    return preds_df

# Page layout
st.title("Chemical Analogues Explorer")

# Input chemical name
chemical_name = st.text_input("Enter a chemical name:", "aspirin")

try:
    # Get chemical info
    inchi = pubchemtools.lookup_chemical_inchi(chemical_name)
    smiles = Chem.MolToSmiles(Chem.MolFromInchi(inchi))
    
    # Draw chemical structure
    mol_img = draw_molecule(smiles)
    if mol_img:
        st.image(mol_img, caption=f"Structure of {chemical_name}")
    
    # Get similar compounds
    similar_compounds = pubchemtools.get_similar_compounds(chemical_name, n_similar=10)
    
    # Get predictions for all compounds
    all_inchis = [inchi] + [pubchemtools.lookup_chemical_inchi(name) for name in similar_compounds]
    all_names = [chemical_name] + similar_compounds
    
    # Get property predictions
    uri_titles = get_uri_titles()
    
    # Property selection dropdowns
    st.subheader("Select Properties to Compare")
    col1, col2 = st.columns(2)
    with col1:
        prop1 = st.selectbox("Property 1:", uri_titles['title'], key='prop1')
    with col2:
        prop2 = st.selectbox("Property 2:", uri_titles['title'], key='prop2')
    
    # Get URIs for selected properties
    uri1 = uri_titles[uri_titles['title'] == prop1]['uri'].iloc[0]
    uri2 = uri_titles[uri_titles['title'] == prop2]['uri'].iloc[0]
    
    # Get predictions for all compounds
    all_predictions = []
    for inchi, name in zip(all_inchis, all_names):
        preds = get_predictions_for_chemical(inchi)
        preds['chemical_name'] = name
        all_predictions.append(preds)
    
    predictions_df = pd.concat(all_predictions)
    
    # Filter for selected properties
    plot_data = predictions_df[predictions_df['uri'].isin([uri1, uri2])]
    plot_data_pivot = plot_data.pivot(index='chemical_name', columns='uri', values='prediction')
    
    # Create scatter plot
    fig = go.Figure()
    
    # Add points for analogues
    fig.add_trace(go.Scatter(
        x=plot_data_pivot[uri1],
        y=plot_data_pivot[uri2],
        mode='markers+text',
        text=plot_data_pivot.index,
        textposition="top center",
        marker=dict(
            size=12,
            color='blue',
            opacity=0.6
        ),
        name='Compounds'
    ))
    
    # Highlight original compound
    orig_data = plot_data_pivot.loc[[chemical_name]]
    fig.add_trace(go.Scatter(
        x=orig_data[uri1],
        y=orig_data[uri2],
        mode='markers+text',
        text=[chemical_name],
        textposition="top center",
        marker=dict(
            size=15,
            color='red',
            opacity=0.8
        ),
        name='Query Compound'
    ))
    
    fig.update_layout(
        title='Property Comparison of Chemical Analogues',
        xaxis_title=prop1,
        yaxis_title=prop2,
        height=600
    )
    
    st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
