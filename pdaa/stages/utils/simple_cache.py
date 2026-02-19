import pandas as pd
import os
import json
import hashlib
from functools import wraps
from pathlib import Path
from typing import Any, Callable

def simple_cache(cache_dir: str | Path, force_refresh: bool = False) -> Callable:
    """
    A decorator that caches function results in JSON files within the specified directory.
    
    Args:
        cache_dir: Directory path where cache files will be stored
        force_refresh: If True, ignore cached results and recompute
        
    Returns:
        Decorated function that implements caching
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_file = cache_path / f"{cache_hash}.json"
            
            # Return cached result if it exists and not forcing refresh
            if not force_refresh and cache_file.exists():
                with open(cache_file, 'r') as f:
                    return json.load(f)
                    
            # Calculate result and cache it
            result = func(*args, **kwargs)
            with open(cache_file, 'w') as f:
                json.dump(result, f)
                
            return result
            
        return wrapper
    return decorator

def simple_cache_df(cache_dir: str | Path, force_refresh: bool = False) -> Callable:
    """
    A decorator that caches pandas DataFrame results in JSON files within the specified directory.
    
    Args:
        cache_dir: Directory path where cache files will be stored
        force_refresh: If True, ignore cached results and recompute
        
    Returns:
        Decorated function that implements caching for pandas DataFrames
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs) -> Any:
            # Create cache key from function name and arguments
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
            cache_file = cache_path / f"{cache_hash}.json"
            
            # Return cached result if it exists and not forcing refresh
            if not force_refresh and cache_file.exists():
                with open(cache_file, 'r') as f:
                    json_data = json.load(f)
                    return pd.DataFrame.from_dict(json_data)
                    
            # Calculate result and cache it
            result = func(*args, **kwargs)
            with open(cache_file, 'w') as f:
                json.dump(result.to_dict(), f)
                
            return result
            
        return wrapper
    return decorator

def get_cache_hash(func: Callable, *args, **kwargs) -> str:
    """Get a hash key for caching a function call.
    
    Example:
        >>> def my_func(x, y=1):
        ...     return x + y
        >>> hash1 = get_cache_hash(my_func, 5, y=2)
        >>> hash2 = get_cache_hash(my_func, 5, y=2) 
        >>> assert hash1 == hash2  # Same args produce same hash
    """
    cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
    cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
    return cache_hash

def load_cache(cache_dir: str | Path) -> dict:
    """Load all cached results from a directory into a dictionary.
    
    Example:
        >>> cache_dir = Path("cache/my_func")
        >>> cache = load_cache(cache_dir)
        >>> # cache = {"hash1": result1, "hash2": result2, ...}
    """
    cache_path = Path(cache_dir)
    cache_path.mkdir(parents=True, exist_ok=True)
    return {f.stem: json.load(open(f)) for f in cache_path.glob('*.json')}

def lookup_in_cache(loaded_cache: dict, key: str) -> Any:
    """Look up a cached result by key.
    
    Example:
        >>> cache = load_cache("cache/my_func")
        >>> hash_key = get_cache_hash(my_func, 5, y=2)
        >>> result = lookup_in_cache(cache, hash_key)
        >>> if result is None:
        ...     # Cache miss - need to compute result
        ...     result = my_func(5, y=2)
    """
    if key in loaded_cache:
        return loaded_cache[key]
    return None