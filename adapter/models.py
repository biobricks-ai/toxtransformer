"""
Adapter Models

Lightweight models trained on ToxTransformer feature vectors
to predict target properties.
"""

import json
import pickle
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


@dataclass
class AdapterConfig:
    """Configuration for adapter models."""
    model_type: str = "logistic"
    feature_selection: str = "none"  # none, variance, topk, category, semantic
    n_features: int = 1000  # for topk selection
    variance_threshold: float = 0.01  # for variance selection
    selected_tokens: Optional[List[int]] = None  # for category/semantic selection
    class_weight: str = "balanced"
    random_state: int = 42
    # Semantic selection config
    target_property: Optional[str] = None  # target property name for semantic selection
    target_description: Optional[str] = None  # optional description
    semantic_model: str = "gemini-2.0-flash"  # LLM model for semantic selection
    semantic_max_categories: int = 10  # max categories for semantic selection
    semantic_cache_path: Optional[str] = None  # cache path for semantic selections
    # Vertex AI config (alternative to Google AI API)
    use_vertex_ai: bool = False  # use Vertex AI instead of Google AI API
    vertex_project: Optional[str] = None  # GCP project ID
    vertex_location: str = "us-central1"  # GCP region
    extra_params: Dict[str, Any] = field(default_factory=dict)


class BaseAdapter(ABC):
    """
    Abstract base class for adapter models.

    All adapters implement a common interface for training, prediction,
    and persistence.
    """

    def __init__(self, config: Optional[AdapterConfig] = None):
        self.config = config or AdapterConfig()
        self.model = None
        self.scaler: Optional[StandardScaler] = None
        self.feature_selector = None
        self.is_fitted = False
        self.feature_names: Optional[List[str]] = None
        self.selected_feature_indices: Optional[np.ndarray] = None
        self.semantic_selection_result = None  # Stores LLM reasoning for semantic selection

    def _setup_feature_selection(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        """Set up and apply feature selection."""
        if self.config.feature_selection == "none":
            self.selected_feature_indices = np.arange(X.shape[1])
            return X

        elif self.config.feature_selection == "variance":
            self.feature_selector = VarianceThreshold(
                threshold=self.config.variance_threshold
            )
            X_selected = self.feature_selector.fit_transform(X)
            self.selected_feature_indices = self.feature_selector.get_support(indices=True)
            return X_selected

        elif self.config.feature_selection == "topk":
            k = min(self.config.n_features, X.shape[1])
            self.feature_selector = SelectKBest(f_classif, k=k)
            X_selected = self.feature_selector.fit_transform(X, y)
            self.selected_feature_indices = self.feature_selector.get_support(indices=True)
            return X_selected

        elif self.config.feature_selection == "category":
            if self.config.selected_tokens is None:
                raise ValueError("selected_tokens required for category feature selection")
            self.selected_feature_indices = np.array(self.config.selected_tokens)
            return X[:, self.selected_feature_indices]

        elif self.config.feature_selection == "semantic":
            # Use LLM to select semantically relevant features
            if self.config.selected_tokens is not None:
                # Already computed (e.g., from trainer)
                self.selected_feature_indices = np.array(self.config.selected_tokens)
                return X[:, self.selected_feature_indices]

            if self.config.target_property is None:
                raise ValueError("target_property required for semantic feature selection")

            from adapter.semantic_selector import get_semantic_feature_indices

            indices, result = get_semantic_feature_indices(
                target_property=self.config.target_property,
                target_description=self.config.target_description,
                property_tokens=None,  # Will use token indices directly
                model=self.config.semantic_model,
                max_categories=self.config.semantic_max_categories,
                cache_path=self.config.semantic_cache_path,
                use_vertex_ai=self.config.use_vertex_ai,
                vertex_project=self.config.vertex_project,
                vertex_location=self.config.vertex_location,
            )

            self.semantic_selection_result = result
            self.selected_feature_indices = indices
            self.config.selected_tokens = indices.tolist()  # Store for reuse

            return X[:, self.selected_feature_indices]

        else:
            raise ValueError(f"Unknown feature selection: {self.config.feature_selection}")

    def _apply_feature_selection(self, X: np.ndarray) -> np.ndarray:
        """Apply fitted feature selection to new data."""
        if self.config.feature_selection == "none":
            return X

        elif self.config.feature_selection in ("variance", "topk"):
            return self.feature_selector.transform(X)

        elif self.config.feature_selection in ("category", "semantic"):
            return X[:, self.selected_feature_indices]

        return X

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaseAdapter":
        """
        Train the adapter model.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels of shape (n_samples,).

        Returns:
            self
        """
        # Feature selection
        X_selected = self._setup_feature_selection(X, y)

        # Standardize features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X_selected)

        # Train model
        self._fit_model(X_scaled, y)
        self.is_fitted = True

        return self

    @abstractmethod
    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        """Train the underlying model. Implemented by subclasses."""
        pass

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Predicted labels of shape (n_samples,).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X_selected = self._apply_feature_selection(X)
        X_scaled = self.scaler.transform(X_selected)
        return self._predict_model(X_scaled)

    @abstractmethod
    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        """Predict using underlying model. Implemented by subclasses."""
        pass

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities.

        Args:
            X: Feature matrix of shape (n_samples, n_features).

        Returns:
            Probability matrix of shape (n_samples, n_classes).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")

        X_selected = self._apply_feature_selection(X)
        X_scaled = self.scaler.transform(X_selected)
        return self._predict_proba_model(X_scaled)

    @abstractmethod
    def _predict_proba_model(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities using underlying model. Implemented by subclasses."""
        pass

    def save(self, path: Union[str, Path]):
        """Save the adapter model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Serialize semantic selection result if present
        semantic_result_dict = None
        if self.semantic_selection_result is not None:
            semantic_result_dict = {
                "selected_tokens": self.semantic_selection_result.selected_tokens,
                "selected_categories": self.semantic_selection_result.selected_categories,
                "reasoning": self.semantic_selection_result.reasoning,
                "n_selected": self.semantic_selection_result.n_selected,
                "n_total": self.semantic_selection_result.n_total,
            }

        state = {
            "config": self.config.__dict__,
            "model": self.model,
            "scaler": self.scaler,
            "feature_selector": self.feature_selector,
            "is_fitted": self.is_fitted,
            "feature_names": self.feature_names,
            "selected_feature_indices": self.selected_feature_indices,
            "model_type": self.__class__.__name__,
            "semantic_selection_result": semantic_result_dict,
        }

        with open(path, "wb") as f:
            pickle.dump(state, f)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "BaseAdapter":
        """Load an adapter model from disk."""
        with open(path, "rb") as f:
            state = pickle.load(f)

        # Determine correct class
        model_type = state.get("model_type", "LogisticAdapter")
        if model_type == "LogisticAdapter":
            adapter = LogisticAdapter.__new__(LogisticAdapter)
        elif model_type == "XGBoostAdapter":
            adapter = XGBoostAdapter.__new__(XGBoostAdapter)
        else:
            adapter = cls.__new__(cls)

        adapter.config = AdapterConfig(**state["config"])
        adapter.model = state["model"]
        adapter.scaler = state["scaler"]
        adapter.feature_selector = state["feature_selector"]
        adapter.is_fitted = state["is_fitted"]
        adapter.feature_names = state.get("feature_names")
        adapter.selected_feature_indices = state.get("selected_feature_indices")

        # Restore semantic selection result if present
        semantic_dict = state.get("semantic_selection_result")
        if semantic_dict:
            from adapter.semantic_selector import SemanticSelectionResult
            adapter.semantic_selection_result = SemanticSelectionResult(**semantic_dict)
        else:
            adapter.semantic_selection_result = None

        return adapter

    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Get feature importances if available.

        Returns:
            Array of feature importances or None.
        """
        return None


class LogisticAdapter(BaseAdapter):
    """
    Logistic Regression adapter.

    Uses sklearn LogisticRegression with L2 regularization.
    Good for linear relationships and interpretability.
    """

    def __init__(self, config: Optional[AdapterConfig] = None, **kwargs):
        if config is None:
            config = AdapterConfig(model_type="logistic")
        super().__init__(config)

        # Allow overriding via kwargs
        self.C = kwargs.get("C", 1.0)
        self.max_iter = kwargs.get("max_iter", 1000)
        self.solver = kwargs.get("solver", "lbfgs")

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        self.model = LogisticRegression(
            C=self.C,
            max_iter=self.max_iter,
            solver=self.solver,
            class_weight=self.config.class_weight if self.config.class_weight != "none" else None,
            random_state=self.config.random_state,
            n_jobs=-1,
        )
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _predict_proba_model(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        # For binary classification, coef_ has shape (1, n_features)
        return np.abs(self.model.coef_).ravel()


class XGBoostAdapter(BaseAdapter):
    """
    XGBoost adapter.

    Uses XGBClassifier for non-linear relationships.
    Better for complex patterns but less interpretable.
    """

    def __init__(self, config: Optional[AdapterConfig] = None, **kwargs):
        if config is None:
            config = AdapterConfig(model_type="xgboost")
        super().__init__(config)

        # XGBoost params
        self.n_estimators = kwargs.get("n_estimators", 100)
        self.max_depth = kwargs.get("max_depth", 6)
        self.learning_rate = kwargs.get("learning_rate", 0.1)
        self.subsample = kwargs.get("subsample", 0.8)
        self.colsample_bytree = kwargs.get("colsample_bytree", 0.8)

    def _fit_model(self, X: np.ndarray, y: np.ndarray):
        try:
            from xgboost import XGBClassifier
        except ImportError:
            raise ImportError("xgboost is required for XGBoostAdapter. Install with: pip install xgboost")

        # Handle class weights for XGBoost
        scale_pos_weight = None
        if self.config.class_weight == "balanced":
            n_neg = np.sum(y == 0)
            n_pos = np.sum(y == 1)
            if n_pos > 0:
                scale_pos_weight = n_neg / n_pos

        self.model = XGBClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            scale_pos_weight=scale_pos_weight,
            random_state=self.config.random_state,
            use_label_encoder=False,
            eval_metric="logloss",
            n_jobs=-1,
        )
        self.model.fit(X, y)

    def _predict_model(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _predict_proba_model(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(X)

    def get_feature_importance(self) -> Optional[np.ndarray]:
        if self.model is None:
            return None
        return self.model.feature_importances_


def create_adapter(model_type: str, config: Optional[AdapterConfig] = None, **kwargs) -> BaseAdapter:
    """
    Factory function to create adapter models.

    Args:
        model_type: One of "logistic" or "xgboost".
        config: Optional AdapterConfig.
        **kwargs: Model-specific parameters.

    Returns:
        Initialized adapter model.
    """
    if config is None:
        config = AdapterConfig(model_type=model_type)
    else:
        config.model_type = model_type

    if model_type == "logistic":
        return LogisticAdapter(config, **kwargs)
    elif model_type == "xgboost":
        return XGBoostAdapter(config, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}. Choose 'logistic' or 'xgboost'.")
