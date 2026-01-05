"""
Prediction cache using SQLite for fast lookups.

Schema:
- predictions: (inchi_hash, property_token, prediction) - individual predictions
- compound_cache: (inchi_hash, inchi, cached_at) - tracks which compounds are fully cached

The inchi_hash is a 64-bit hash of the InChI string for fast lookups.
"""
import sqlite3
import hashlib
import threading
import logging
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from contextlib import contextmanager

logger = logging.getLogger(__name__)


def inchi_to_hash(inchi: str) -> int:
    """Convert InChI to 63-bit signed hash for SQLite compatibility."""
    # SQLite INTEGER is signed 64-bit, so we use 63 bits to stay positive
    h = int(hashlib.sha256(inchi.encode()).hexdigest()[:15], 16)
    return h


class PredictionCache:
    """Thread-safe SQLite prediction cache."""

    def __init__(self, db_path: str = "cache/predictions.sqlite"):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._init_db()

    def _get_conn(self) -> sqlite3.Connection:
        """Get thread-local database connection."""
        if not hasattr(self._local, 'conn'):
            self._local.conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=30.0
            )
            self._local.conn.execute("PRAGMA journal_mode=WAL")
            self._local.conn.execute("PRAGMA synchronous=NORMAL")
            self._local.conn.execute("PRAGMA cache_size=-64000")  # 64MB cache
        return self._local.conn

    def _init_db(self):
        """Initialize database schema."""
        conn = self._get_conn()
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS predictions (
                inchi_hash INTEGER NOT NULL,
                property_token INTEGER NOT NULL,
                prediction REAL NOT NULL,
                PRIMARY KEY (inchi_hash, property_token)
            ) WITHOUT ROWID;

            CREATE TABLE IF NOT EXISTS compound_cache (
                inchi_hash INTEGER PRIMARY KEY,
                inchi TEXT NOT NULL,
                cached_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            ) WITHOUT ROWID;

            CREATE INDEX IF NOT EXISTS idx_predictions_hash
                ON predictions(inchi_hash);
        """)
        conn.commit()

    def get_prediction(self, inchi: str, property_token: int) -> Optional[float]:
        """Get a single cached prediction."""
        inchi_hash = inchi_to_hash(inchi)
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT prediction FROM predictions WHERE inchi_hash = ? AND property_token = ?",
            (inchi_hash, property_token)
        )
        row = cursor.fetchone()
        return row[0] if row else None

    def get_all_predictions(self, inchi: str) -> Optional[Dict[int, float]]:
        """Get all cached predictions for a compound, or None if not fully cached."""
        inchi_hash = inchi_to_hash(inchi)
        conn = self._get_conn()

        # Check if compound is fully cached
        cursor = conn.execute(
            "SELECT 1 FROM compound_cache WHERE inchi_hash = ?",
            (inchi_hash,)
        )
        if not cursor.fetchone():
            return None

        # Get all predictions
        cursor = conn.execute(
            "SELECT property_token, prediction FROM predictions WHERE inchi_hash = ?",
            (inchi_hash,)
        )
        return {row[0]: row[1] for row in cursor.fetchall()}

    def cache_prediction(self, inchi: str, property_token: int, prediction: float):
        """Cache a single prediction."""
        inchi_hash = inchi_to_hash(inchi)
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO predictions (inchi_hash, property_token, prediction) VALUES (?, ?, ?)",
            (inchi_hash, property_token, prediction)
        )
        conn.commit()

    def cache_all_predictions(self, inchi: str, predictions: Dict[int, float]):
        """Cache all predictions for a compound (marks as fully cached)."""
        inchi_hash = inchi_to_hash(inchi)
        conn = self._get_conn()

        # Insert all predictions
        conn.executemany(
            "INSERT OR REPLACE INTO predictions (inchi_hash, property_token, prediction) VALUES (?, ?, ?)",
            [(inchi_hash, prop, pred) for prop, pred in predictions.items()]
        )

        # Mark compound as fully cached
        conn.execute(
            "INSERT OR REPLACE INTO compound_cache (inchi_hash, inchi) VALUES (?, ?)",
            (inchi_hash, inchi)
        )
        conn.commit()

    def cache_batch(self, batch: List[Tuple[str, int, float]]):
        """Cache a batch of predictions efficiently."""
        conn = self._get_conn()
        conn.executemany(
            "INSERT OR REPLACE INTO predictions (inchi_hash, property_token, prediction) VALUES (?, ?, ?)",
            [(inchi_to_hash(inchi), prop, pred) for inchi, prop, pred in batch]
        )
        conn.commit()

    def mark_compound_cached(self, inchi: str):
        """Mark a compound as fully cached."""
        inchi_hash = inchi_to_hash(inchi)
        conn = self._get_conn()
        conn.execute(
            "INSERT OR REPLACE INTO compound_cache (inchi_hash, inchi) VALUES (?, ?)",
            (inchi_hash, inchi)
        )
        conn.commit()

    def is_compound_cached(self, inchi: str) -> bool:
        """Check if a compound is fully cached."""
        inchi_hash = inchi_to_hash(inchi)
        conn = self._get_conn()
        cursor = conn.execute(
            "SELECT 1 FROM compound_cache WHERE inchi_hash = ?",
            (inchi_hash,)
        )
        return cursor.fetchone() is not None

    def get_stats(self) -> Dict:
        """Get cache statistics."""
        conn = self._get_conn()

        pred_count = conn.execute("SELECT COUNT(*) FROM predictions").fetchone()[0]
        compound_count = conn.execute("SELECT COUNT(*) FROM compound_cache").fetchone()[0]

        # Get database file size
        db_size = self.db_path.stat().st_size if self.db_path.exists() else 0

        return {
            "predictions_cached": pred_count,
            "compounds_fully_cached": compound_count,
            "database_size_mb": db_size / (1024 * 1024),
            "database_path": str(self.db_path)
        }

    def get_uncached_compounds(self, inchis: List[str]) -> List[str]:
        """Filter to only compounds that aren't fully cached."""
        conn = self._get_conn()
        hashes = {inchi_to_hash(inchi): inchi for inchi in inchis}

        placeholders = ",".join("?" * len(hashes))
        cursor = conn.execute(
            f"SELECT inchi_hash FROM compound_cache WHERE inchi_hash IN ({placeholders})",
            list(hashes.keys())
        )
        cached_hashes = {row[0] for row in cursor.fetchall()}

        return [inchi for h, inchi in hashes.items() if h not in cached_hashes]
