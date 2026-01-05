import json
import threading
import pandas as pd
import numpy as np
import cvae.models.multitask_encoder as mte
import cvae.spark_helpers as H
import torch
import torch.nn
import sqlite3
import itertools
import logging
import os
from torch.utils.data import DataLoader, TensorDataset
from dataclasses import dataclass
from tqdm import tqdm
from dataclasses import dataclass, asdict
from typing import List, Optional, Dict, Tuple, Callable

from flask_cvae.prediction_cache import PredictionCache

# Set up logging
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s:%(message)s',
                    handlers=[logging.StreamHandler()])

# Use GPU if available, otherwise CPU
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Performance settings
USE_FP16 = torch.cuda.is_available()  # Use half precision on GPU
USE_COMPILE = False  # torch.compile has issues with dynamic shapes
MI_THRESHOLD = 0.07  # Minimum MI to include property as context


@dataclass
class Category:
    category: str
    reason: str
    strength: str

@dataclass
class Property:
    property_token: int
    source: str
    title: str
    metadata: dict
    categories: list[Category]

@dataclass
class Prediction:
    inchi: str
    property_token: int
    property: Property
    value: float

class Predictor:

    def __init__(self, cache_path: str = "cache/predictions.sqlite", use_cache: bool = True):
        self.dburl = 'brick/cvae.sqlite'
        self.dblock = threading.Lock()
        model_path = os.environ.get("MODEL_PATH", "cache/full_train/logs/models/final_model_V3/best_loss")

        logging.info(f"Loading model from {model_path}...")
        self.model = mte.MultitaskEncoder.load(model_path).to(DEVICE)
        self.tokenizer = self.model.tokenizer
        self.model.eval()

        # Performance optimizations
        if USE_FP16:
            logging.info("Enabling FP16 inference")
            self.model = self.model.half()

        if USE_COMPILE and hasattr(torch, 'compile'):
            logging.info("Compiling model with torch.compile...")
            self.model = torch.compile(self.model, mode="reduce-overhead")

        # Initialize prediction cache
        self.use_cache = use_cache
        self.cache = PredictionCache(cache_path) if use_cache else None
        if use_cache:
            stats = self.cache.get_stats()
            logging.info(f"Prediction cache initialized: {stats['compounds_fully_cached']:,} compounds, {stats['predictions_cached']:,} predictions")

        conn = sqlite3.connect(self.dburl)
        conn.row_factory = sqlite3.Row
        self.all_property_tokens = [int(r['property_token']) for r in conn.execute("SELECT DISTINCT property_token FROM property")]
        self.property_map = self.build_property_map()
        conn.close()

        # Load mutual information lookup for context selection
        self.mi_lookup = self._load_mi_lookup()
        logging.info(f"Loaded MI lookup for {len(self.mi_lookup)} properties")

    def _load_mi_lookup(self) -> Dict[int, List[Tuple[int, float]]]:
        """
        Load pairwise mutual information and build lookup.
        Returns: target_property -> [(context_property, mi_score), ...] sorted by MI descending.
        """
        mi_path = os.environ.get("MI_PATH", "cache/token_information/pairwise_mi.parquet")

        try:
            mi_df = pd.read_parquet(mi_path)
            logging.info(f"Loaded MI data from {mi_path}: {len(mi_df)} pairs")
        except Exception as e:
            logging.warning(f"Could not load MI data from {mi_path}: {e}. Using random context selection.")
            return {}

        # Filter by threshold
        filtered = mi_df[mi_df['mi'] > MI_THRESHOLD].copy()

        # Create bidirectional mapping (MI is symmetric)
        forward = filtered[['prop1', 'prop2', 'mi']].rename(
            columns={'prop1': 'target', 'prop2': 'context'}
        )
        backward = filtered[['prop2', 'prop1', 'mi']].rename(
            columns={'prop2': 'target', 'prop1': 'context'}
        )
        combined = pd.concat([forward, backward], ignore_index=True)

        # Build lookup dict
        lookup = {}
        for target, group in combined.groupby('target'):
            # Sort by MI descending
            sorted_contexts = group.sort_values('mi', ascending=False)
            lookup[int(target)] = [
                (int(row['context']), float(row['mi']))
                for _, row in sorted_contexts.iterrows()
            ]

        return lookup

    def _select_context_by_mi(self, known_props: Dict[int, int], target_prop: int, max_context: int) -> List[Tuple[int, int]]:
        """
        Select context properties ranked by mutual information with target.

        Args:
            known_props: Dict of property_token -> value for known properties
            target_prop: Target property to predict
            max_context: Maximum number of context properties

        Returns:
            List of (property_token, value) tuples, ordered by MI with target
        """
        if target_prop not in self.mi_lookup:
            # Fallback to arbitrary selection if no MI data
            return [(p, v) for p, v in list(known_props.items())[:max_context] if p != target_prop]

        # Get context properties ranked by MI with target
        ranked_contexts = self.mi_lookup[target_prop]

        selected = []
        for context_prop, mi_score in ranked_contexts:
            if context_prop in known_props and context_prop != target_prop:
                selected.append((context_prop, known_props[context_prop]))
                if len(selected) >= max_context:
                    break

        return selected

    def build_property_map(self):
        with self.dblock:
            conn = sqlite3.connect(self.dburl)
            conn.row_factory = lambda cursor, row: dict((cursor.description[i][0], value) for i, value in enumerate(row))
            cursor = conn.cursor()
            cursor.execute("""
                SELECT p.property_token, p.title, p.data as metadata, s.source, c.category, pc.reason, pc.strength
                FROM property p
                INNER JOIN property_category pc ON p.property_id = pc.property_id
                INNER JOIN category c ON pc.category_id = c.category_id
                INNER JOIN source s ON p.source_id = s.source_id
            """)
            res = cursor.fetchall()

            # Group results by property_token
            property_map = {}
            for property_token, group in itertools.groupby(res, key=lambda x: x['property_token']):
                group_list = list(group)
                categories = [Category(category=r['category'], reason=r['reason'], strength=r['strength'])
                            for r in group_list]

                property = Property(property_token=int(property_token),
                                  title=group_list[0]['title'],
                                  metadata=json.loads(group_list[0]['metadata']),
                                  source=group_list[0]['source'],
                                  categories=categories)

                property_map[int(property_token)] = property

            return property_map

    def _get_known_properties(self, inchi, category=None):
        """Get known property-value pairs for a molecule from the database."""
        conn = sqlite3.connect(self.dburl)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        query = """
        SELECT DISTINCT prop.property_token, act.value
        FROM activity act
        INNER JOIN property prop ON act.property_id = prop.property_id
        WHERE inchi = ?"""

        params = [inchi]
        if category is not None:
            query += """ AND prop.property_id IN (
                SELECT property_id FROM property_category pc
                INNER JOIN category cat ON pc.category_id = cat.category_id
                WHERE cat.category = ?
            )"""
            params.append(category)

        cursor.execute(query, params)
        res = [dict(row) for row in cursor.fetchall()]
        conn.close()

        if len(res) == 0:
            return {}

        # Return as dict: property_token -> value
        return {int(r['property_token']): int(r['value']) for r in res}

    def _encode_selfies(self, inchi):
        """Convert InChI to SELFIES token tensor."""
        smiles = H.inchi_to_smiles_safe(inchi)
        selfies = H.smiles_to_selfies_safe(smiles)
        selfies_tokens = self.tokenizer.selfies_tokenizer.selfies_to_indices(selfies)
        return torch.LongTensor(selfies_tokens)

    def predict_property(self, inchi, property_token, max_context=10) -> Prediction:
        """
        Predict a single property for a molecule using MI-ranked context.

        Args:
            inchi: InChI string of the molecule
            property_token: Property token ID to predict
            max_context: Maximum number of context properties to use
        """
        property_token = int(property_token)

        if property_token not in self.all_property_tokens:
            logging.error(f"Property token {property_token} is not valid")
            return None

        # Check cache first
        if self.use_cache:
            cached_value = self.cache.get_prediction(inchi, property_token)
            if cached_value is not None:
                token_property = self.property_map.get(property_token, None)
                return Prediction(inchi=inchi, property_token=property_token,
                                property=token_property, value=cached_value)

        # Encode molecule
        selfies_tensor = self._encode_selfies(inchi)

        # Get known properties for context
        known_props = self._get_known_properties(inchi)

        # Select context using MI ranking (key improvement!)
        context_props = self._select_context_by_mi(known_props, property_token, max_context)

        # Build properties tensor: [context_props..., target_prop]
        props_list = [p for p, v in context_props] + [property_token]
        properties = torch.LongTensor(props_list)

        # Build values tensor: [context_values..., 0 for target placeholder]
        values_list = [v for p, v in context_props] + [0]
        values = torch.LongTensor(values_list)

        # Build mask: all True
        mask = torch.ones(len(props_list), dtype=torch.bool)

        # Add batch dimension and move to device
        selfies_batch = selfies_tensor.unsqueeze(0).to(DEVICE)
        properties_batch = properties.unsqueeze(0).to(DEVICE)
        values_batch = values.unsqueeze(0).to(DEVICE)
        mask_batch = mask.unsqueeze(0).to(DEVICE)

        # Convert to half precision if enabled
        if USE_FP16:
            selfies_batch = selfies_batch
            properties_batch = properties_batch
            values_batch = values_batch

        # Run inference
        with torch.no_grad():
            with torch.amp.autocast('cuda', enabled=USE_FP16):
                logits = self.model(selfies_batch, properties_batch, values_batch, mask_batch)

            # Get prediction at the last position (target)
            target_logits = logits[0, -1]  # [2] for binary classification
            prob = torch.softmax(target_logits.float(), dim=-1)[1].item()  # Probability of class 1

        # Cache the prediction
        if self.use_cache:
            self.cache.cache_prediction(inchi, property_token, prob)

        token_property = self.property_map.get(property_token, None)
        prediction = Prediction(inchi=inchi, property_token=property_token, property=token_property, value=prob)
        return prediction

    def _precompute_all_contexts(self, known_props: Dict[int, int], max_context: int) -> Dict[int, List[Tuple[int, int]]]:
        """
        Pre-compute MI-ranked context for all target properties at once.
        Much faster than calling _select_context_by_mi in a loop.
        """
        # Convert known_props to set for O(1) lookup
        known_set = set(known_props.keys())

        all_contexts = {}
        for target_prop in self.all_property_tokens:
            if target_prop not in self.mi_lookup:
                # No MI data - use empty context (faster than arbitrary selection)
                all_contexts[target_prop] = []
                continue

            # Get context properties ranked by MI with target
            ranked_contexts = self.mi_lookup[target_prop]

            selected = []
            for context_prop, mi_score in ranked_contexts:
                if context_prop in known_set and context_prop != target_prop:
                    selected.append((context_prop, known_props[context_prop]))
                    if len(selected) >= max_context:
                        break

            all_contexts[target_prop] = selected

        return all_contexts

    def predict_all_properties(self, inchi, max_context=10, batch_size=4096,
                                progress_callback: Optional[Callable[[float], None]] = None) -> list[Prediction]:
        """
        Predict all properties for a molecule using MI-ranked context.
        Optimized version with pre-computed contexts and larger batches.

        Args:
            inchi: InChI string of the molecule
            max_context: Maximum number of context properties to use
            batch_size: Number of properties to predict per batch
            progress_callback: Optional callback(progress: float) called with 0.0-1.0
        """
        # Check if fully cached
        if self.use_cache:
            cached_preds = self.cache.get_all_predictions(inchi)
            if cached_preds is not None:
                logging.info(f"Cache hit for {inchi[:50]}... ({len(cached_preds)} predictions)")
                predictions = []
                for prop_token, value in cached_preds.items():
                    token_property = self.property_map.get(prop_token, None)
                    predictions.append(Prediction(
                        inchi=inchi, property_token=prop_token,
                        property=token_property, value=value
                    ))
                return predictions

        # Encode molecule once
        selfies_tensor = self._encode_selfies(inchi)
        selfies_len = selfies_tensor.size(0)

        # Get known properties for context
        known_props = self._get_known_properties(inchi)

        # Pre-compute all contexts at once (major speedup)
        all_contexts = self._precompute_all_contexts(known_props, max_context)

        predictions = []
        predictions_to_cache = {}

        # Pre-allocate the selfies tensor expanded for batch (reused across batches)
        # This avoids repeated memory allocation

        # Process in batches
        for i in range(0, len(self.all_property_tokens), batch_size):
            batch_tokens = self.all_property_tokens[i:i+batch_size]
            batch_len = len(batch_tokens)

            # Build batch data using pre-computed contexts
            props_lists = []
            values_lists = []
            seq_lengths = []

            for target_prop in batch_tokens:
                ctx = all_contexts[target_prop]
                props_list = [p for p, v in ctx] + [target_prop]
                values_list = [v for p, v in ctx] + [0]
                props_lists.append(props_list)
                values_lists.append(values_list)
                seq_lengths.append(len(props_list))

            max_prop_len = max(seq_lengths)

            # Create padded tensors directly (more efficient than stacking)
            properties_padded = torch.zeros(batch_len, max_prop_len, dtype=torch.long)
            values_padded = torch.zeros(batch_len, max_prop_len, dtype=torch.long)
            mask_padded = torch.zeros(batch_len, max_prop_len, dtype=torch.bool)

            for j, (props, vals, seq_len) in enumerate(zip(props_lists, values_lists, seq_lengths)):
                properties_padded[j, :seq_len] = torch.tensor(props, dtype=torch.long)
                values_padded[j, :seq_len] = torch.tensor(vals, dtype=torch.long)
                mask_padded[j, :seq_len] = True

            # Expand selfies for batch
            selfies_padded = selfies_tensor.unsqueeze(0).expand(batch_len, -1)

            # Move to device
            selfies_padded = selfies_padded.to(DEVICE)
            properties_padded = properties_padded.to(DEVICE)
            values_padded = values_padded.to(DEVICE)
            mask_padded = mask_padded.to(DEVICE)

            # Run inference with autocast for FP16
            with torch.no_grad():
                with torch.amp.autocast('cuda', enabled=USE_FP16):
                    logits = self.model(selfies_padded, properties_padded, values_padded, mask_padded)

                # Vectorized extraction of predictions
                last_positions = torch.tensor(seq_lengths, device=DEVICE) - 1
                batch_indices = torch.arange(batch_len, device=DEVICE)

                # Get logits at last positions for all samples at once
                target_logits = logits[batch_indices, last_positions]  # [batch, 2]
                probs = torch.softmax(target_logits.float(), dim=-1)[:, 1]  # [batch]
                probs_list = probs.cpu().tolist()

                for j, prob in enumerate(probs_list):
                    prop_token = batch_tokens[j]
                    token_property = self.property_map.get(prop_token, None)
                    pred = Prediction(inchi=inchi, property_token=prop_token, property=token_property, value=prob)
                    predictions.append(pred)
                    predictions_to_cache[prop_token] = prob

            # Report progress after each batch
            if progress_callback:
                progress = min(1.0, (i + batch_len) / len(self.all_property_tokens))
                progress_callback(progress)

        # Cache all predictions
        if self.use_cache and predictions_to_cache:
            self.cache.cache_all_predictions(inchi, predictions_to_cache)

        return predictions
