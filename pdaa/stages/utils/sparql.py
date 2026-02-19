import hashlib
import pandas as pd
import requests
from rdflib import Graph, Namespace
from typing import List, Dict, Any
import pathlib

class Query:
    
    def __init__(self, graph:Graph, cachedir:pathlib.Path=None):
        self.select_vars: List[str] = []
        self.where_clauses: List[str] = []
        self.graph = graph
        self.column_types: Dict[str, Any] = {}
        self.cachedir = cachedir
        if cachedir is not None:
            self.cachedir.mkdir(parents=True, exist_ok=True)

    def select(self, *variables: str) -> 'Query':
        """Add variables to select in the SPARQL query"""
        if self.select_vars is None:
            raise ValueError("Cannot select variables twice")
        self.select_vars.extend(variables)
        return self

    def select_typed(self, type_mapping: Dict[str, Any]) -> 'Query':
        """
        Set pandas data types for variables in the query results
        
        Args:
            type_mapping: Dictionary mapping variable names to Python/pandas types
                         e.g. {'age': int, 'name': str, 'score': float}
        """
        self.column_types = type_mapping
        self.select_vars = list(type_mapping.keys())
        return self
        
    def where(self, triple_pattern: str) -> 'Query':
        """Add a where clause to the SPARQL query"""
        self.where_clauses.append(triple_pattern)
        return self
    
    def limit(self, limit: int) -> 'Query':
        """Add a limit clause to the SPARQL query"""
        self.limit_clause = f"LIMIT {limit}"
        return self
    
    def build_query(self) -> str:
        """Build the complete SPARQL query string"""
        select_clause = "SELECT " + " ".join(f"?{var}" for var in self.select_vars)
        where_clause = "WHERE { " + " . ".join(self.where_clauses) + " }"
        if hasattr(self, 'limit_clause'):
            where_clause += f" {self.limit_clause}"
        return f"{select_clause}\n{where_clause}"
    
    def cache_execute(self, force_refresh:bool=False) -> pd.DataFrame:
        """Execute the query and cache results as a pandas DataFrame with proper types
        
        Args:
            cache_path: Path to save/load cached results
            
        Returns:
            DataFrame containing query results
        """
        query_str = self.build_query()
        query_hash = hashlib.md5(query_str.encode()).hexdigest()
        cache_path = self.cachedir / f"{query_hash}.parquet"
        
        # Return cached results if they exist
        if cache_path.exists() and not force_refresh:
            df = pd.read_parquet(cache_path)
            return df
            
        # Execute query and cache results
        df = self.execute()
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(cache_path)
        return df
    
    def execute(self) -> pd.DataFrame:
        """Execute the query and return results as a pandas DataFrame with proper types"""
        query_str = self.build_query()
        results = self.graph.query(query_str)
        
        # Convert results to DataFrame
        df = pd.DataFrame(results.bindings).map(str)
        df.columns = [str(c) for c in df.columns]
        df = df[self.select_vars]
        
        # Apply type casting if types were specified
        if hasattr(self, 'column_types'):
            for col, dtype in self.column_types.items():
                if col in df.columns:
                    df[col] = df[col].astype(dtype)
                    
        return df

# import sys
# sys.path.append('./')
# import stages.utils.pdaa as pdaa

# q = Query(pdaa.pdaa_graph) \
#     .select_typed({'uri': str, 'proptoken': int, 'token': str}) \
#     .where('?proptoken <http://purl.org/dc/elements/1.1/has_identifier> ?uri') \
#     .where('?proptoken a <http://toxindex.com/ontology/predicted_property>') \
#     .where('?proptoken rdf:value ?token') \
#     .execute()

# self=Query(endpoint='http://localhost:8000/sparql/aop') \
#     .select('aop', 'mie', 'ao') \
#     .where('?aop aop:has_molecular_initiating_event ?mie') \
#     .where('?aop aop:has_adverse_outcome ?ao') \
#     .execute()