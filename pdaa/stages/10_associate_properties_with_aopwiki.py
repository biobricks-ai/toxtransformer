import sys
sys.path.append('./')

import stages.utils.openai as openai_utils
import stages.utils.sparql as sparql
import stages.utils.toxindex as toxindex

import json
import faiss
import shutil
import sqlite3
import rdflib
import pathlib
import functools
import biobricks
import subprocess
import numpy as np
import pandas as pd

from tqdm import tqdm

cachedir = pathlib.Path('cache/associate_properties_with_aopwiki')
cachedir.mkdir(parents=True, exist_ok=True)

tqdm.pandas()

# outs: [aop_titles, aop_descriptions, aop_abstracts, key_events, membership, graph]
## BUILD AOPWIKI RDF ========================================================
if not pathlib.Path('./hdtworkdir/out.nt').exists():
    sr = functools.partial(subprocess.run, shell=True)
    sr("git clone https://github.com/rdfhdt/hdt-cpp.git")
    sr('docker build -t hdt hdt-cpp/.')

    # start the container
    hdtworkdir = pathlib.Path('./hdtworkdir')
    hdtworkdir.exists() and shutil.rmtree(hdtworkdir)
    hdtworkdir.mkdir(parents=True, exist_ok=True)

    sr(f'docker rm -f hdt || true')
    sr(f'docker run -d --name hdt -v $(pwd)/hdtworkdir:/workdir hdt tail -f /dev/null')

    # ADD AOPWikiRDF-Genes.hdt AOPWikiRDF.hdt TO DOCKER CONTAINER
    aopwiki = biobricks.Brick.Resolve('aopwikirdf-kg').path() / 'brick'
    for hdt_file in aopwiki.glob('*.hdt'):
        sr(f'docker cp {hdt_file.resolve()} hdt:/workdir/{hdt_file.name}')

    hdtcmd = lambda cmd: sr(f'docker exec -it hdt /bin/bash -c "{cmd}"')
    hdtcmd(f"hdt2rdf /workdir/AOPWikiRDF.hdt /workdir/out.nt")

## QUERY AOPWIKI RDF ========================================================
graph = rdflib.Graph()
graph.namespace_manager.bind('aop', rdflib.Namespace('http://aopkb.org/aop_ontology#'))
graph.namespace_manager.bind('dcterms', rdflib.Namespace('http://purl.org/dc/terms/'))
graph.parse('./hdtworkdir/out.nt', format='nt')

aopwiki_uri_text = sparql.Query(graph) \
    .select('uri', 'variable', 'value') \
    .where('?uri a ?type') \
    .where('FILTER(?type = aop:KeyEvent || ?type = aop:AdverseOutcomePathway)') \
    .where('?uri ?variable ?value') \
    .where('FILTER (?variable = dc:title || ?variable = rdfs:label || ?variable = dc:description)') \
    .execute()

# endregion

# TODO: we need better URIs for bindingdb
# outs: 
#   - propvars - property_token, title, data, source
#   - proptokens - uri, property_token
# region GET CHEMPROP-TRANSFORMER PROPERTIES ========================================================
with sqlite3.connect(biobricks.assets('chemprop-transformer').cvae_sqlite) as con:
    rawprops = pd.read_sql_query("SELECT property_token, title, data, s.source FROM property p INNER JOIN source s ON p.source_id = s.source_id", con)
    rawprops['property_token'] = rawprops['property_token'].astype(int)
    rawprops['data'] = rawprops['data'].map(lambda x: json.loads(x))
    property_categories = pd.read_sql_query("SELECT p.property_id, p.property_token, c.category_id, c.category FROM property_category inner join category c on property_category.category_id = c.category_id inner join property p on property_category.property_id = p.property_id", con)
    property_categories['category_id'] = property_categories['category_id'].astype(int)

# ice example uri https://ice.ntp.niehs.nih.gov/api/v1/curves?assay=BSK_LPS_TNFa_down
ctice = rawprops[rawprops['source'] == 'ice'].reset_index()
ctice['uri'] = ctice['data'].progress_apply(lambda x: f"https://ice.ntp.niehs.nih.gov/api/v1/curves?assay={x['Assay'].replace(' ', '_')}")

# binding db
ctbindingdb = rawprops[rawprops['source'] == 'bindingdb'].reset_index()
ctbindingdb['uri'] = ctbindingdb['data'].progress_apply(lambda x: x['Link to Target in BindingDB'])

# pubchem
ctpubchem = rawprops[rawprops['source'] == 'pubchem'].reset_index()
mkaid = lambda aid: f"https://identifiers.org/pubchem.bioassay:{aid}"
ctpubchem['uri'] = ctpubchem['data'].progress_apply(lambda x: mkaid(int(x['aid'])))

# TODO need better uris for chembl maybe something like https://www.ebi.ac.uk/chembl/explore/target/CHEMBL688612
ctchembl = rawprops[rawprops['source'] == 'chembl'].reset_index()
mkassayid = lambda aid: f"https://identifiers.org/chembl.target:{aid}"
ctchembl['uri'] = ctchembl['data'].progress_apply(lambda x: mkassayid(x['chembl_id']))

ctprops = pd.concat([ctbindingdb, ctpubchem, ctchembl, ctice], ignore_index=True)
ctprops['data'] = ctprops['data'].map(lambda x: json.dumps(x))

# associate uris with property_token
proptokens = ctprops[['uri','title','property_token','data']].drop_duplicates()
proptokens.to_csv(cachedir / 'proptokens.csv', index=False)

# build text values
propvars = ctprops[['uri','title','data']]
propvars = pd.melt(ctprops, id_vars=['uri'], value_vars=['title','data'], var_name='variable', value_name='value')
propvars = propvars.drop_duplicates()

# endregion

# region ASSOCIATE PROPERTIES WITH AOPWIKI PATHWAYS ========================================================
embed_df = pd.concat([propvars, aopwiki_uri_text], ignore_index=True)
embed_df = embed_df[['uri','variable','value']]
embed_df = embed_df.dropna().drop_duplicates()

# there shouldn't be any duplicates
assert embed_df.shape[0] == propvars.shape[0] + aopwiki_uri_text.shape[0]

truncate = lambda text: text[:10000] if len(text) > 10000 else text
embed_df['embedding'] = embed_df['value'].apply(truncate).progress_apply(openai_utils.embed)
embed_df['embedding'] = embed_df['embedding'].map(lambda x: x / np.linalg.norm(x))
embed_df.to_csv(cachedir / 'embed_df.csv', index=False)

# Normalize embeddings for cosine similarity
embeddings = np.vstack(embed_df['embedding'].values)

# Create and save FAISS index
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
faiss.write_index(index, (cachedir / 'faiss_index_cosine.index').as_posix())

# endregion

# region CREATE RDF ========================================================
# write a new ntriples that adds:
# 1. ctprops and their property_token,source, title, and data
# 2. similarity entities that link ctprops and key events
import pandas as pd
from rdflib import Literal, URIRef, XSD, RDFS, RDF
proptokens = pd.read_csv(cachedir / 'proptokens.csv')
embed_df = pd.read_csv(cachedir / 'embed_df.csv')

proptokens[proptokens['uri'] == testuri]
embed_df[embed_df['uri'] == testuri]
DCTERMS = rdflib.Namespace('http://purl.org/dc/elements/1.1/')

simgraph = rdflib.Graph()
simgraph.parse('./hdtworkdir/out.nt', format='nt')
print(f"there are {len(simgraph)} triples in the graph")

## CREATE PROPERTY_TOKEN ENTITIES FROM CHEMPROP-TRANSFORMER with
## --- RDF.type: toxindex:predicted_property
## --- has_identifier:  <the external uri>
## --- purl:title: from the gpt4 generated title
## --- rdfs:label: the numeric property_token
added_tuple_set = set()
def add_tuple(tuple):
    if tuple in added_tuple_set:
        return
    added_tuple_set.add(tuple)
    simgraph.add(tuple)

for ind, rawuri, title, property_token, data in tqdm(list(proptokens.itertuples())):
    uri = URIRef(rawuri)
    token_uri = toxindex.property_token_to_uri(property_token)
    add_tuple((token_uri, RDF.type, toxindex.predicted_property_class))
    add_tuple((token_uri, DCTERMS.term('has_identifier'), uri))
    add_tuple((token_uri, DCTERMS.term('title'), Literal(title)))
    add_tuple((token_uri, DCTERMS.term('description'), Literal(data)))
    add_tuple((token_uri, RDF.value, Literal(int(property_token), datatype=XSD.integer)))

# link uris to faiss index
for i, uri in tqdm(list(enumerate(embed_df['uri']))):
    faiss_token = toxindex.faiss_index_to_uri(i)
    faiss_value = Literal(int(i),datatype=XSD.integer)
    add_tuple((faiss_token, DCTERMS.term('has_identifier'), URIRef(uri)))
    add_tuple((faiss_token, RDF.value, faiss_value))
    add_tuple((faiss_token, RDF.type, toxindex.faissclass))

print(f"there are {len(simgraph)} triples in the graph")

# Commit the changes to the graph
simgraph.commit()
simgraph.serialize(destination=cachedir / 'simgraph.nt', format='nt')
simgraph.close()

# add to blazegraph
# delete the pdaa namespace if it exists
import requests

# delete the namespace if it exists
response = requests.delete(f'http://localhost:9999/blazegraph/namespace/pdaa')

# create the pdaa namespace
namespace_url = 'http://localhost:9999/blazegraph/namespace'
headers = {'Content-Type': 'application/xml'}
namespace_properties = '''<?xml version="1.0" encoding="UTF-8" standalone="no"?>
<!DOCTYPE properties SYSTEM "http://java.sun.com/dtd/properties.dtd">
<properties>
    <entry key="com.bigdata.rdf.store.AbstractTripleStore.textIndex">true</entry>
    <entry key="com.bigdata.rdf.store.AbstractTripleStore.axiomsClass">com.bigdata.rdf.axioms.NoAxioms</entry>
    <entry key="com.bigdata.rdf.sail.isolatableIndices">false</entry>
    <entry key="com.bigdata.rdf.sail.truthMaintenance">false</entry>
    <entry key="com.bigdata.rdf.store.AbstractTripleStore.justify">false</entry>
    <entry key="com.bigdata.rdf.sail.namespace">pdaa</entry>
</properties>'''
response = requests.post(namespace_url, headers=headers, data=namespace_properties)
response.raise_for_status()

# upload the simgraph.nt to blazegraph
with open(cachedir / 'simgraph.nt', 'rb') as f:
    headers = {'Content-Type': 'application/x-turtle'}
    response = requests.post(f"{namespace_url}/pdaa/sparql", headers=headers, data=f)
response.raise_for_status()
# endregion