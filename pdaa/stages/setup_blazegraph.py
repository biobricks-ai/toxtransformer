import requests

def upload_triples_to_blazegraph(file_path, endpoint_url):
    """
    Upload RDF triples to Blazegraph via POST request
    
    Args:
        file_path: Path to .nt file containing triples
        endpoint_url: URL of Blazegraph SPARQL endpoint
    """
    with open(file_path, 'rb') as f:
        headers = {'Content-Type': 'application/x-turtle'}
        response = requests.post(endpoint_url, headers=headers, data=f)
        
        if response.status_code == 200:
            print(f"Successfully uploaded triples from {file_path}")
        else:
            print(f"Error uploading triples: {response.status_code}")
            print(response.text)

# Example usage:
# upload_triples_to_blazegraph(
#     'data.nt',
#     'http://localhost:9999/blazegraph/namespace/mygraph/sparql'
# )

# first make the namespace
# Create namespace for PDAA data
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

response = requests.post(f'{namespace_url}', headers=headers, data=namespace_properties)


upload_triples_to_blazegraph(
    'cache/associate_properties_with_aopwiki/simgraph.nt',
    'http://localhost:9999/blazegraph/namespace/pdaa/sparql'
)

# Test query to verify data was loaded
test_query = """
SELECT ?s ?p ?o 
WHERE { 
    ?s ?p ?o 
} 
LIMIT 10
"""

headers = {'Accept': 'application/sparql-results+json'}
response = requests.get(
    'http://localhost:9999/blazegraph/namespace/pdaa/sparql',
    params={'query': test_query},
    headers=headers
)

if response.status_code == 200:
    results = response.json()
    print("\nTest query results:")
    for binding in results['results']['bindings']:
        print(f"Subject: {binding['s']['value']}")
        print(f"Predicate: {binding['p']['value']}")
        print(f"Object: {binding['o']['value']}")
        print()
else:
    print(f"Error running test query: {response.status_code}")
    print(response.text)

# Create RDFLib graph connected to Blazegraph endpoint
from rdflib import Graph
from rdflib.plugins.stores.sparqlstore import SPARQLStore

# Create a SPARQL store pointing to the Blazegraph endpoint
store = SPARQLStore('http://localhost:9999/blazegraph/namespace/pdaa/sparql')

# Create the graph using the SPARQL store
graph = Graph(store=store)
res = graph.query(test_query)

res.bindings[0]

# delete the pdaa namespace
response = requests.delete(f'{namespace_url}/pdaa')
if response.status_code == 200:
    print("Successfully deleted namespace 'pdaa'")
else:
    print(f"Error deleting namespace: {response.status_code}")
    print(response.text)