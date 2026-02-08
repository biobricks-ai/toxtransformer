"""
ToxTransformer Adapter Module

Uses ToxTransformer's 6,647 property predictions as a feature vector to train
simple models for target properties not directly in the training data.

Supports LLM-based semantic feature selection to reduce overfitting by selecting
only toxicologically relevant features based on the target property.
"""

from adapter.feature_extractor import ToxTransformerFeatureExtractor
from adapter.models import BaseAdapter, LogisticAdapter, XGBoostAdapter, AdapterConfig
from adapter.trainer import AdapterTrainer, AdapterResult
from adapter.evaluator import AdapterEvaluator
from adapter.semantic_selector import (
    SemanticFeatureSelector,
    SemanticSelectionResult,
    ToxTransformerMetadataLoader,
    get_semantic_feature_indices,
)

__all__ = [
    "ToxTransformerFeatureExtractor",
    "BaseAdapter",
    "LogisticAdapter",
    "XGBoostAdapter",
    "AdapterConfig",
    "AdapterTrainer",
    "AdapterResult",
    "AdapterEvaluator",
    "SemanticFeatureSelector",
    "SemanticSelectionResult",
    "ToxTransformerMetadataLoader",
    "get_semantic_feature_indices",
]
