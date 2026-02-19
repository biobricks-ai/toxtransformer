from rdflib import URIRef

faissclass = URIRef('http://toxindex.com/ontology/faiss_index')
def faiss_index_to_uri(index):
    return URIRef(f"{faissclass}/faiss_index{index}")

predicted_property_class = URIRef('http://toxindex.com/ontology/predicted_property')
def property_token_to_uri(property_token):
    return URIRef(f"{predicted_property_class}/predicted_property{property_token}")