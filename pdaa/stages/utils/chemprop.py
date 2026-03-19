import aiohttp
import asyncio
import requests
import time
import logging
import os
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GCP endpoint for toxtransformer
CHEMPROP_BASE_URL = os.environ.get("TOXTRANSFORMER_API_URL", "http://136.111.102.10:6515")

_iap_token_cache: dict = {"token": None, "expires_at": 0.0}

def _get_iap_headers() -> dict:
    """Return Authorization header with IAP ID token if IAP_CLIENT_ID is configured."""
    client_id = os.environ.get("TOXTRANSFORMER_IAP_CLIENT_ID", "")
    if not client_id:
        return {}

    now = time.time()
    if _iap_token_cache["token"] and now < _iap_token_cache["expires_at"]:
        return {"Authorization": f"Bearer {_iap_token_cache['token']}"}

    try:
        # Use GCP metadata server directly — reliable in Cloud Run
        metadata_url = (
            "http://metadata.google.internal/computeMetadata/v1/"
            f"instance/service-accounts/default/identity"
            f"?audience={client_id}&format=full"
        )
        resp = requests.get(
            metadata_url,
            headers={"Metadata-Flavor": "Google"},
            timeout=5
        )
        resp.raise_for_status()
        token = resp.text.strip()
        logger.info(f"Fetched IAP token for audience {client_id[:20]}... (len={len(token)})")
        # ID tokens are valid for 1 hour; refresh 5 minutes early
        _iap_token_cache["token"] = token
        _iap_token_cache["expires_at"] = now + 3300
        return {"Authorization": f"Bearer {token}"}
    except Exception as e:
        logger.warning(f"Could not fetch IAP token from metadata server: {e}")
        return {}

def make_safe(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            return {"result": None, "error": str(e)}
    return wrapper


@retry(stop=stop_after_attempt(5), wait=wait_exponential(multiplier=1, min=2, max=10))
def get_chemprop_prediction(inchi: str, property_token: str) -> dict:
    """Synchronous single property prediction."""
    base_url = f"{CHEMPROP_BASE_URL}/predict"
    params = {"property_token": property_token, "inchi": inchi}
    response = requests.get(base_url, params=params, headers=_get_iap_headers())
    response.raise_for_status()
    return response.json()


def get_chemprop_prediction_safe(inchi: str, property_token: str, retries: int = 5, delay: int = 2) -> dict:
    return make_safe(get_chemprop_prediction)(inchi, property_token, retries, delay)


def chemprop_predict_all(inchi: str, timeout: float = 120.0) -> list[dict]:
    """
    Predict all properties using the /predict_all endpoint.

    Args:
        inchi: InChI string of the molecule
        timeout: Request timeout in seconds (default 120.0)

    Returns:
        List of prediction dicts with 'inchi', 'property_token', 'value' keys
    """
    url = f"{CHEMPROP_BASE_URL}/predict_all"
    params = {"inchi": inchi}

    logger.info(f"Requesting predictions for {inchi[:50]}...")
    response = requests.get(url, params=params, headers=_get_iap_headers(), timeout=timeout)
    response.raise_for_status()
    results = response.json()

    # Transform to expected format
    predictions = []
    for item in results:
        if item.get("value") is not None:  # Skip null predictions
            predictions.append({
                "inchi": item["inchi"],
                "property_token": item["property_token"],
                "value": item["value"]
            })

    logger.info(f"Got {len(predictions)} predictions for {inchi[:50]}...")
    return predictions


async def chemprop_predict_all_async(inchi: str, timeout: float = 1200.0) -> list[dict]:
    """
    Predict all properties via jobs.toxindex.com /api/v1/run/toxtransformer.

    Converts InChI to SMILES, posts to the synchronous jobs API, and parses
    the response into the standard PDAA format.

    Args:
        inchi: InChI string of the molecule
        timeout: Request timeout in seconds (default 1200.0)

    Returns:
        List of dicts with 'inchi', 'property_token', 'value' keys
    """
    from rdkit import Chem
    from rdkit.Chem.inchi import MolFromInchi

    mol = MolFromInchi(inchi)
    if mol is None:
        raise ValueError(f"Cannot parse InChI: {inchi[:80]}")
    smiles = Chem.MolToSmiles(mol)

    url = f"{CHEMPROP_BASE_URL}/api/v1/run/toxtransformer"
    iap_headers = _get_iap_headers()

    logger.info(f"Submitting to jobs.toxindex.com for {smiles[:40]}...")
    async with aiohttp.ClientSession(headers=iap_headers) as session:
        async with session.post(
            url,
            json={"smiles": smiles},
            timeout=aiohttp.ClientTimeout(total=timeout)
        ) as response:
            response.raise_for_status()
            data = await response.json()

    predictions = data.get("prediction", {}).get("predictions", [])
    result = []
    for pred in predictions:
        response_inchi = pred.get("inchi", inchi)
        for prop in pred.get("properties", []):
            val = prop.get("value")
            if val is not None:
                result.append({
                    "inchi": response_inchi,
                    "property_token": prop["property_token"],
                    "value": val,
                })
    logger.info(f"Got {len(result)} predictions for {smiles[:40]}")
    return result


def predict_all_simple(inchi: str, timeout: float = 120.0) -> list[tuple]:
    """
    Simple synchronous prediction using /predict_all endpoint.

    Args:
        inchi: InChI string of the molecule
        timeout: Request timeout in seconds

    Returns:
        List of (inchi, property_token, prediction) tuples
    """
    url = f"{CHEMPROP_BASE_URL}/predict_all"
    params = {"inchi": inchi}

    response = requests.get(url, params=params, headers=_get_iap_headers(), timeout=timeout)
    response.raise_for_status()
    results = response.json()

    # Transform to (inchi, token, prediction) format
    transformed = []
    for item in results:
        if item.get("value") is not None:  # Skip null predictions
            transformed.append((
                item["inchi"],
                item["property_token"],
                item["value"]
            ))

    logger.info(f"Got {len(transformed)} predictions for {inchi[:50]}...")
    return transformed


async def get_chemprop_prediction_async(inchi: str, property_token: str, retries: int = 5, delay: int = 2) -> dict:
    """
    Async single property prediction with retry logic.

    Args:
        inchi: InChI string of the molecule
        property_token: Property token ID for the prediction
        retries: Number of retry attempts
        delay: Delay (in seconds) between retries

    Returns:
        dict: Dictionary containing 'result' and 'error' keys
    """
    base_url = f"{CHEMPROP_BASE_URL}/predict"
    params = {
        "property_token": property_token,
        "inchi": inchi
    }

    async with aiohttp.ClientSession(headers=_get_iap_headers()) as session:
        for attempt in range(1, retries + 1):
            try:
                async with session.get(base_url, params=params) as response:
                    response.raise_for_status()
                    result = await response.json()
                    return {"result": result, "error": None}
            except Exception as e:
                if attempt < retries:
                    await asyncio.sleep(delay)
                else:
                    return {"result": None, "error": f"Failed after {retries} retries: {str(e)}"}
