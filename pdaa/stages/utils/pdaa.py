import asyncio
import faiss
import numpy as np
import pandas as pd
import pathlib
import rdflib
from rdflib import URIRef
from rdflib.plugins.stores.sparqlstore import SPARQLStore
import sqlite3
import threading
from tqdm import tqdm

import stages.utils.chemprop as chemprop
import stages.utils.openai as openai_utils
import stages.utils.sparql as sparql

brickdir = pathlib.Path('brick')
sqlite_lock = threading.Lock()

cachedir = pathlib.Path('cache') / 'util' / 'pdaa'
cachedir.mkdir(parents=True, exist_ok=True)

# Check if predictions database exists
db_path = brickdir / 'predictions.sqlite'
db_available = db_path.exists()

if not db_available:
    print(f"⚠️  Predictions database not found at {db_path}")
    print("   Running in API-only mode (no local caching)")

# uri faiss index
faiss_index = faiss.read_index('cache/associate_properties_with_aopwiki/faiss_index_cosine.index')

# Create a SPARQL store pointing to the Blazegraph endpoint
pdaa_graph = rdflib.Graph(store=SPARQLStore('http://localhost:9999/blazegraph/namespace/pdaa/sparql'))
pdaa_graph.namespace_manager.bind('aop', rdflib.Namespace('http://aopkb.org/aop_ontology#'))
pdaa_graph.namespace_manager.bind('toxindex', rdflib.Namespace('http://toxindex.com/ontology/'))
pdaa_graph.namespace_manager.bind('dcterms', rdflib.Namespace('http://purl.org/dc/elements/1.1/'))
pdaa_graph_cache = cachedir / 'pdaa_graph'

uri_faissindex = sparql.Query(pdaa_graph, pdaa_graph_cache) \
    .select_typed({'uri': str, 'index': int}) \
    .where('?faissindex <http://purl.org/dc/elements/1.1/has_identifier> ?uri') \
    .where('?faissindex a <http://toxindex.com/ontology/faiss_index>') \
    .where('?faissindex rdf:value ?index') \
    .cache_execute()

uri_faissindex['embedding'] = uri_faissindex['index'].map(lambda i: faiss_index.reconstruct(i))

# Load mappings for molecular initiating events (MIE) and adverse outcomes (AO)
aop_mie_ao = sparql.Query(pdaa_graph, pdaa_graph_cache) \
    .select('aop', 'mie', 'ao') \
    .where('?aop aop:has_adverse_outcome ?ao') \
    .where('?aop aop:has_molecular_initiating_event ?mie') \
    .cache_execute()

# Fetch URIs linked to property tokens
# TODO some predicted_properties have multiple tokens
proptoken_uris = sparql.Query(pdaa_graph, pdaa_graph_cache) \
    .select_typed({'uri': str, 'proptoken': str, 'token': int, 'title': str}) \
    .where('?proptoken <http://purl.org/dc/elements/1.1/has_identifier> ?uri') \
    .where('?proptoken a <http://toxindex.com/ontology/predicted_property>') \
    .where('?proptoken rdf:value ?token') \
    .where('?proptoken <http://purl.org/dc/elements/1.1/title> ?title') \
    .cache_execute() \
    .groupby('uri').first().reset_index()

def lookup_predictions(inchi_tok_pairs):
    """Look up cached predictions from SQLite database (if available)."""
    if not db_available:
        return []
    
    with sqlite_lock:
        with sqlite3.connect(db_path) as conn:
            results = []
            for inchi, property_token in inchi_tok_pairs:
                cursor = conn.execute("""SELECT inchi, CAST(property_token AS INTEGER) as property_token, positive_prediction FROM predictions 
                                      WHERE inchi = ? AND property_token = ?""", (inchi, property_token))
                result = cursor.fetchone()
                if result is not None:
                    results.append((inchi, property_token, result[2]))
            return results

def add_predictions(predictions, lock):
    """Add predictions to SQLite database (if available)."""
    if not db_available:
        return
    
    with lock:
        with sqlite3.connect(db_path) as conn:
            for inchi, property_token, positive_prediction in predictions:
                conn.execute('INSERT INTO predictions (inchi, property_token, positive_prediction) VALUES (?, ?, ?)', 
                           (inchi, property_token, positive_prediction))

def is_missing(inchi_list):
    """Check which InChIs are missing from the cache. Returns all if no database."""
    if not db_available:
        return inchi_list  # All missing if no database
    
    with sqlite_lock:
        with sqlite3.connect(db_path) as conn:
            missing = []
            for inchi in inchi_list:
                cursor = conn.execute("SELECT COUNT(*) FROM predictions WHERE inchi = ?", (inchi,))
                count = cursor.fetchone()[0]
                if count == 0:
                    missing.append(inchi)
            return missing

def predict_all_properties_with_sqlite_cache(inchi_list):
    """Get predictions, using SQLite cache if available, otherwise API only."""
    missing_inchi = is_missing(inchi_list)
    preds = []

    # Get predictions for missing InChIs from API using async endpoint to avoid lock contention
    import asyncio
    async def get_all_predictions():
        tasks = [chemprop.chemprop_predict_all_async(inchi) for inchi in missing_inchi]
        return await asyncio.gather(*tasks)

    if missing_inchi:
        results = asyncio.run(get_all_predictions())
        for api_preds in results:
            preds.extend(api_preds)
    
    # Convert to format for database
    preds_db = [(fullpred['inchi'], int(fullpred['property_token']), fullpred['value']) for fullpred in preds]
    
    # Add to database if available
    if db_available:
        add_predictions(preds_db, sqlite_lock)
    
    # Look up cached predictions for non-missing InChIs
    non_missing_inchi = [inchi for inchi in inchi_list if inchi not in missing_inchi]
    for tok in proptoken_uris['token']:
        preds_db.extend(lookup_predictions([(inchi, tok) for inchi in non_missing_inchi]))
    
    return preds_db

async def async_predict_all(inchi,semaphore):
    async with semaphore:
        return chemprop.chemprop_predict_all(inchi)

def get_uri_similars(uris, target_uris, top_k_to_search=100):
    """Use cached FAISS index to find similar URIs."""
    target_faiss_indices = uri_faissindex[uri_faissindex['uri'].isin(target_uris)]['index'].values
    search_uris = uri_faissindex[uri_faissindex['uri'].isin(uris)]
    
    simdf = []
    for uri, embedding in zip(search_uris['uri'], search_uris['embedding']):
        embedding_norm = embedding / np.linalg.norm(embedding)
        embedding_2d = embedding_norm[np.newaxis, :]
        
        distances, indices = faiss_index.search(embedding_2d, min(top_k_to_search, uri_faissindex.shape[0]))
        
        indices_in_target = np.isin(indices[0], target_faiss_indices)
        filtered_indices = indices[0][indices_in_target]
        filtered_distances = distances[0][indices_in_target]
        
        for idx, dist in zip(filtered_indices, filtered_distances):
            matched_uri = uri_faissindex[uri_faissindex['index'] == idx]['uri'].values[0]
            simdf.append({'uri_search': uri, 'uri_match': matched_uri, 'similarity': 1 - dist})
    
    return pd.DataFrame(simdf).sort_values('similarity', ascending=False)

faissclass = URIRef('http://toxindex.com/ontology/faiss_index')
def faiss_index_to_uri(index):
    return URIRef(f"{faissclass}/faiss_index{index}")

predicted_property_class = URIRef('http://toxindex.com/ontology/predicted_property')
def property_token_to_uri(property_token):
    return URIRef(f"{predicted_property_class}/predicted_property{property_token}")

def get_prompt_similars(prompt, target_uris, top_k_to_search=100):
    """Search for URIs similar to a text prompt using OpenAI embeddings + FAISS."""
    prompt_embedding = np.array(openai_utils.embed(prompt))[np.newaxis, :] 
    prompt_embedding = prompt_embedding / np.linalg.norm(prompt_embedding)
    
    distances, indices = faiss_index.search(prompt_embedding, min(top_k_to_search, uri_faissindex.shape[0]))
    
    matched_uris = []
    for idx, dist in zip(indices[0], distances[0]):
        matched_uri = uri_faissindex[uri_faissindex['index'] == idx]['uri'].values[0]
        if matched_uri in target_uris:
            matched_uris.append({'uri': matched_uri, 'similarity': 1 - dist})
    
    return pd.DataFrame(matched_uris).sort_values('similarity', ascending=False)
