import rdflib
from rdflib import plugin, URIRef

class SQLiteGraph(rdflib.Graph):
    """
    A graph that is stored in a sqlite database.
    Example usage:
        >>> # Create a new SQLite-backed graph
        >>> graph = SQLiteGraph("sqlite:///example.sqlite")
        >>> 
        >>> # Add some triples
        >>> graph.add((URIRef("http://example.org/s"),
        ...           URIRef("http://example.org/p"), 
        ...           URIRef("http://example.org/o")))
        >>>
        >>> # Query the graph
        >>> for s,p,o in graph.triples((None, None, None)):
        ...     print(s,p,o)
        >>>
        >>> # Commit changes and close
        >>> graph.commit()
        >>> graph.close()
    """
    def __init__(self, path, identifier):
        store = plugin.get('SQLAlchemy', rdflib.store.Store)()
        super().__init__(store, identifier=identifier)
        self.open(f"sqlite:///{path}", create=True)
