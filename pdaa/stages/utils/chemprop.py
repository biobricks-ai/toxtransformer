import aiohttp
import asyncio
import requests
import time
import logging
from tenacity import retry, stop_after_attempt, wait_exponential
from functools import wraps

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# GCP endpoint for toxtransformer
CHEMPROP_BASE_URL = "http://136.111.102.10:6515"

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
    response = requests.get(base_url, params=params)
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
    response = requests.get(url, params=params, timeout=timeout)
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


async def chemprop_predict_all_async(inchi: str, poll_interval: float = 2.0, timeout: float = 300.0) -> list[dict]:
    """
    Async version: Predict all properties using the async job queue.

    Args:
        inchi: InChI string of the molecule
        poll_interval: Seconds between status polls (default 2.0)
        timeout: Maximum seconds to wait for job completion (default 300.0)

    Returns:
        List of prediction dicts with 'inchi', 'property_token', 'value' keys
    """
    jobs_url = f"{CHEMPROP_BASE_URL}/jobs"

    async with aiohttp.ClientSession() as session:
        # Submit job
        job_data = {
            "job_type": "predict_all",
            "inchi": inchi
        }

        async with session.post(jobs_url, json=job_data) as response:
            response.raise_for_status()
            job_info = await response.json()

        job_id = job_info["job_id"]
        logger.info(f"Submitted job {job_id} for {inchi[:50]}... (queue position: {job_info.get('queue_position', '?')})")

        # Poll for completion
        status_url = f"{jobs_url}/{job_id}"
        start_time = time.time()

        while True:
            elapsed = time.time() - start_time
            if elapsed > timeout:
                raise TimeoutError(f"Job {job_id} timed out after {timeout}s")

            async with session.get(status_url) as status_response:
                status_response.raise_for_status()
                status = await status_response.json()

            job_status = status["status"]
            progress = status.get("progress", 0)

            if job_status == "completed":
                logger.info(f"Job {job_id} completed in {elapsed:.1f}s")
                return status["result"]

            elif job_status == "failed":
                error = status.get("error", "Unknown error")
                raise RuntimeError(f"Job {job_id} failed: {error}")

            elif job_status in ("pending", "processing"):
                eta = status.get("estimated_seconds")
                if eta:
                    logger.debug(f"Job {job_id}: {job_status} ({progress*100:.0f}%), ETA: {eta:.0f}s")
                await asyncio.sleep(poll_interval)

            else:
                raise RuntimeError(f"Unknown job status: {job_status}")


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

    response = requests.get(url, params=params, timeout=timeout)
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

    async with aiohttp.ClientSession() as session:
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
