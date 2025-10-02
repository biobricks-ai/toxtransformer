import pathlib
import pickle
import hashlib
import functools
import inspect

def simplecache(cache_dir):
    cache_dir = pathlib.Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                func_source = inspect.getsource(func)
            except OSError:
                func_source = ""  # fallback if source is not available

            key_bytes = pickle.dumps((func_source, args, kwargs))
            key_hash = hashlib.sha1(key_bytes).hexdigest()
            cache_path = cache_dir / f"{func.__name__}_{key_hash}.pkl"

            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    return pickle.load(f)

            result = func(*args, **kwargs)
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
            return result

        return wrapper
    return decorator
