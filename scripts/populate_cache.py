#!/usr/bin/env python3
"""
Populate the prediction cache with all compounds from the training data.

Usage:
    # Populate cache for all compounds (will take ~60 days on single GPU)
    PYTHONPATH=./ python scripts/populate_cache.py

    # Populate cache for first N compounds (for testing)
    PYTHONPATH=./ python scripts/populate_cache.py --limit 1000

    # Use specific GPU
    CUDA_VISIBLE_DEVICES=1 PYTHONPATH=./ python scripts/populate_cache.py

    # Resume from checkpoint
    PYTHONPATH=./ python scripts/populate_cache.py --resume
"""
import argparse
import logging
import sqlite3
import time
import json
from pathlib import Path
from datetime import datetime, timedelta

import torch
from tqdm import tqdm

import cvae.models.multitask_encoder as mte
import cvae.spark_helpers as H
from flask_cvae.prediction_cache import PredictionCache

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


class CachePopulator:
    def __init__(self, cache_path: str = "cache/predictions.sqlite",
                 checkpoint_path: str = "cache/populate_checkpoint.json"):
        self.cache = PredictionCache(cache_path)
        self.checkpoint_path = Path(checkpoint_path)

        # Load model
        logger.info("Loading model...")
        self.model = mte.MultitaskEncoder.load(
            "cache/train_multitask_transformer_parallel/logs/split_0/models/me_roundrobin_property_dropout_V3/best_loss"
        ).to(DEVICE)
        self.model.eval()
        self.tokenizer = self.model.tokenizer
        logger.info(f"Model loaded on {DEVICE}")

        # Get all property tokens
        conn = sqlite3.connect('brick/cvae.sqlite')
        self.all_property_tokens = [
            int(r[0]) for r in conn.execute("SELECT DISTINCT property_token FROM property")
        ]
        conn.close()
        logger.info(f"Found {len(self.all_property_tokens)} property tokens")

    def get_all_compounds(self) -> list[str]:
        """Get all unique InChI strings from the database."""
        logger.info("Fetching all compounds from database...")
        conn = sqlite3.connect('brick/cvae.sqlite')
        cursor = conn.execute("SELECT DISTINCT inchi FROM activity")
        compounds = [row[0] for row in cursor.fetchall() if row[0]]
        conn.close()
        logger.info(f"Found {len(compounds):,} unique compounds")
        return compounds

    def load_checkpoint(self) -> int:
        """Load checkpoint and return the index to resume from."""
        if self.checkpoint_path.exists():
            with open(self.checkpoint_path) as f:
                data = json.load(f)
                return data.get('last_index', 0) + 1
        return 0

    def save_checkpoint(self, index: int, stats: dict):
        """Save progress checkpoint."""
        data = {
            'last_index': index,
            'timestamp': datetime.now().isoformat(),
            'stats': stats
        }
        with open(self.checkpoint_path, 'w') as f:
            json.dump(data, f, indent=2)

    def predict_all_for_compound(self, inchi: str, batch_size: int = 512) -> dict[int, float]:
        """Predict all properties for a single compound."""
        try:
            smiles = H.inchi_to_smiles_safe(inchi)
            selfies = H.smiles_to_selfies_safe(smiles)
            selfies_tokens = torch.LongTensor(
                self.tokenizer.selfies_tokenizer.selfies_to_indices(selfies)
            ).to(DEVICE)
        except Exception as e:
            logger.warning(f"Failed to encode {inchi[:50]}...: {e}")
            return {}

        predictions = {}

        for i in range(0, len(self.all_property_tokens), batch_size):
            batch_props = self.all_property_tokens[i:i+batch_size]
            bs = len(batch_props)

            selfies_batch = selfies_tokens.unsqueeze(0).repeat(bs, 1)
            properties_batch = torch.LongTensor([[p] for p in batch_props]).to(DEVICE)
            values_batch = torch.zeros(bs, 1, dtype=torch.long, device=DEVICE)
            mask_batch = torch.ones(bs, 1, dtype=torch.bool, device=DEVICE)

            with torch.no_grad():
                logits = self.model(selfies_batch, properties_batch, values_batch, mask_batch)
                probs = torch.softmax(logits[:, 0], dim=-1)[:, 1].cpu().numpy()

            for prop, prob in zip(batch_props, probs):
                predictions[prop] = float(prob)

        return predictions

    def populate(self, limit: int = None, resume: bool = False, batch_size: int = 512):
        """Populate the cache with predictions for all compounds."""
        compounds = self.get_all_compounds()

        if limit:
            compounds = compounds[:limit]
            logger.info(f"Limited to {limit} compounds")

        # Filter out already cached compounds
        uncached = self.cache.get_uncached_compounds(compounds)
        logger.info(f"Compounds to process: {len(uncached):,} (skipping {len(compounds) - len(uncached):,} cached)")

        if not uncached:
            logger.info("All compounds already cached!")
            return

        start_idx = 0
        if resume:
            start_idx = self.load_checkpoint()
            if start_idx > 0:
                logger.info(f"Resuming from index {start_idx}")
                uncached = uncached[start_idx:]

        # Progress tracking
        start_time = time.time()
        total = len(uncached)

        pbar = tqdm(enumerate(uncached), total=total, desc="Populating cache")
        for i, inchi in pbar:
            try:
                predictions = self.predict_all_for_compound(inchi, batch_size)
                if predictions:
                    self.cache.cache_all_predictions(inchi, predictions)
            except Exception as e:
                logger.error(f"Error processing compound {i}: {e}")
                continue

            # Update progress
            elapsed = time.time() - start_time
            rate = (i + 1) / elapsed
            remaining = (total - i - 1) / rate if rate > 0 else 0
            eta = timedelta(seconds=int(remaining))

            pbar.set_postfix({
                'rate': f'{rate:.2f}/s',
                'ETA': str(eta),
            })

            # Checkpoint every 1000 compounds
            if (i + 1) % 1000 == 0:
                stats = self.cache.get_stats()
                self.save_checkpoint(start_idx + i, stats)
                logger.info(f"Checkpoint saved: {stats['compounds_fully_cached']:,} compounds cached")

        # Final stats
        stats = self.cache.get_stats()
        logger.info(f"Cache population complete!")
        logger.info(f"  Compounds cached: {stats['compounds_fully_cached']:,}")
        logger.info(f"  Predictions cached: {stats['predictions_cached']:,}")
        logger.info(f"  Database size: {stats['database_size_mb']:.1f} MB")


def main():
    parser = argparse.ArgumentParser(description="Populate prediction cache")
    parser.add_argument('--limit', type=int, help="Limit number of compounds")
    parser.add_argument('--resume', action='store_true', help="Resume from checkpoint")
    parser.add_argument('--batch-size', type=int, default=512, help="Batch size for predictions")
    parser.add_argument('--cache-path', default="cache/predictions.sqlite", help="Cache database path")
    args = parser.parse_args()

    populator = CachePopulator(cache_path=args.cache_path)
    populator.populate(
        limit=args.limit,
        resume=args.resume,
        batch_size=args.batch_size
    )


if __name__ == "__main__":
    main()
