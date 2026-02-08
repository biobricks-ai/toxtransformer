"""
ToxTransformer Feature Extractor

Wraps ToxTransformer API to extract 6,647-dimensional feature vectors
from molecular predictions.
"""

import hashlib
import json
import logging
import os
import sqlite3
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import requests
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)


@dataclass
class FeatureExtractionResult:
    """Result of feature extraction for a single molecule."""
    inchi: str
    features: np.ndarray
    property_tokens: List[int]
    success: bool
    error: Optional[str] = None


class ToxTransformerFeatureExtractor:
    """
    Extracts ToxTransformer predictions as feature vectors.

    Uses the ToxTransformer API to get all 6,647 property predictions
    for a molecule, which serve as features for downstream adapters.
    """

    DEFAULT_API_URL = "http://136.111.102.10:6515"
    NUM_PROPERTIES = 6647

    def __init__(
        self,
        api_url: Optional[str] = None,
        cache_path: Optional[str] = None,
        timeout: int = 300,
        max_workers: int = 4,
    ):
        """
        Initialize the feature extractor.

        Args:
            api_url: ToxTransformer API URL. Defaults to configured server.
            cache_path: Path to SQLite cache file. If None, caching is disabled.
            timeout: Request timeout in seconds.
            max_workers: Max parallel requests for batch extraction.
        """
        self.api_url = api_url or os.environ.get(
            "TOXTRANSFORMER_API_URL", self.DEFAULT_API_URL
        )
        self.timeout = timeout
        self.max_workers = max_workers

        # Initialize cache
        self.cache_path = cache_path
        self._cache_lock = threading.Lock()
        if cache_path:
            self._init_cache()

        # Property token order (will be populated on first extraction)
        self._property_tokens: Optional[List[int]] = None
        self._token_to_idx: Optional[Dict[int, int]] = None

    def _init_cache(self):
        """Initialize SQLite cache for storing predictions."""
        Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.cache_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS feature_cache (
                    inchi_hash TEXT PRIMARY KEY,
                    inchi TEXT,
                    features BLOB,
                    property_tokens TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_feature_cache_hash ON feature_cache(inchi_hash)"
            )

    def _hash_inchi(self, inchi: str) -> str:
        """Create a hash of the InChI for cache lookup."""
        return hashlib.sha256(inchi.encode()).hexdigest()[:32]

    def _get_cached(self, inchi: str) -> Optional[Tuple[np.ndarray, List[int]]]:
        """Get cached features for an InChI."""
        if not self.cache_path:
            return None

        with self._cache_lock:
            with sqlite3.connect(self.cache_path) as conn:
                cursor = conn.execute(
                    "SELECT features, property_tokens FROM feature_cache WHERE inchi_hash = ?",
                    (self._hash_inchi(inchi),),
                )
                row = cursor.fetchone()
                if row:
                    features = np.frombuffer(row[0], dtype=np.float32)
                    tokens = json.loads(row[1])
                    return features, tokens
        return None

    def _set_cached(self, inchi: str, features: np.ndarray, tokens: List[int]):
        """Cache features for an InChI."""
        if not self.cache_path:
            return

        with self._cache_lock:
            with sqlite3.connect(self.cache_path) as conn:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO feature_cache (inchi_hash, inchi, features, property_tokens)
                    VALUES (?, ?, ?, ?)
                    """,
                    (
                        self._hash_inchi(inchi),
                        inchi,
                        features.astype(np.float32).tobytes(),
                        json.dumps(tokens),
                    ),
                )

    def _call_api(self, inchi: str) -> Dict:
        """Call ToxTransformer API to get all predictions."""
        url = f"{self.api_url}/predict_all"
        response = requests.get(url, params={"inchi": inchi}, timeout=self.timeout)
        response.raise_for_status()
        return response.json()

    def _predictions_to_features(
        self, predictions: List[Dict]
    ) -> Tuple[np.ndarray, List[int]]:
        """
        Convert API predictions to feature vector.

        Returns:
            Tuple of (features array, property token list)
        """
        # Sort by property_token to ensure consistent ordering
        sorted_preds = sorted(predictions, key=lambda x: x["property_token"])

        tokens = [p["property_token"] for p in sorted_preds]
        features = np.array([p["value"] for p in sorted_preds], dtype=np.float32)

        return features, tokens

    def extract(self, inchi: str) -> FeatureExtractionResult:
        """
        Extract feature vector for a single molecule.

        Args:
            inchi: InChI string of the molecule.

        Returns:
            FeatureExtractionResult with 6,647-dimensional feature vector.
        """
        # Check cache
        cached = self._get_cached(inchi)
        if cached is not None:
            features, tokens = cached
            return FeatureExtractionResult(
                inchi=inchi,
                features=features,
                property_tokens=tokens,
                success=True,
            )

        # Call API
        try:
            predictions = self._call_api(inchi)
            features, tokens = self._predictions_to_features(predictions)

            # Update property token order on first extraction
            if self._property_tokens is None:
                self._property_tokens = tokens
                self._token_to_idx = {t: i for i, t in enumerate(tokens)}

            # Cache result
            self._set_cached(inchi, features, tokens)

            return FeatureExtractionResult(
                inchi=inchi,
                features=features,
                property_tokens=tokens,
                success=True,
            )

        except requests.RequestException as e:
            logging.error(f"API error for {inchi[:50]}...: {e}")
            return FeatureExtractionResult(
                inchi=inchi,
                features=np.zeros(self.NUM_PROPERTIES, dtype=np.float32),
                property_tokens=[],
                success=False,
                error=str(e),
            )

        except Exception as e:
            logging.error(f"Extraction error for {inchi[:50]}...: {e}")
            return FeatureExtractionResult(
                inchi=inchi,
                features=np.zeros(self.NUM_PROPERTIES, dtype=np.float32),
                property_tokens=[],
                success=False,
                error=str(e),
            )

    def batch_extract(
        self,
        inchis: List[str],
        show_progress: bool = True,
    ) -> np.ndarray:
        """
        Extract feature vectors for multiple molecules.

        Args:
            inchis: List of InChI strings.
            show_progress: Show progress bar.

        Returns:
            Feature matrix of shape (n_molecules, 6647).
        """
        results = []
        failed = []

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = {executor.submit(self.extract, inchi): i for i, inchi in enumerate(inchis)}

            iterator = as_completed(futures)
            if show_progress:
                iterator = tqdm(iterator, total=len(inchis), desc="Extracting features")

            for future in iterator:
                idx = futures[future]
                result = future.result()
                results.append((idx, result))
                if not result.success:
                    failed.append(inchis[idx])

        if failed:
            logging.warning(f"Failed to extract features for {len(failed)} molecules")

        # Sort by original order
        results.sort(key=lambda x: x[0])

        # Stack into matrix
        feature_matrix = np.vstack([r.features for _, r in results])

        return feature_matrix

    def get_property_tokens(self) -> Optional[List[int]]:
        """Get the list of property tokens in feature order."""
        return self._property_tokens

    def get_feature_names(self) -> List[str]:
        """Get feature names (property tokens as strings)."""
        if self._property_tokens is None:
            return [f"prop_{i}" for i in range(self.NUM_PROPERTIES)]
        return [f"prop_{t}" for t in self._property_tokens]

    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        if not self.cache_path:
            return {"cached": 0, "cache_enabled": False}

        with sqlite3.connect(self.cache_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM feature_cache")
            count = cursor.fetchone()[0]

        return {"cached": count, "cache_enabled": True, "cache_path": self.cache_path}
