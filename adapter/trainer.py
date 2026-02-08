"""
Adapter Trainer

Training pipeline with k-fold cross-validation for adapter models.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold

from adapter.feature_extractor import ToxTransformerFeatureExtractor
from adapter.models import (
    AdapterConfig,
    BaseAdapter,
    LogisticAdapter,
    XGBoostAdapter,
    create_adapter,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)


@dataclass
class FoldResult:
    """Results from a single CV fold."""
    fold: int
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_train: int
    n_val: int
    n_pos_train: int
    n_pos_val: int


@dataclass
class AdapterResult:
    """Results from adapter training."""
    model: BaseAdapter
    fold_results: List[FoldResult]
    mean_auc: float
    std_auc: float
    mean_accuracy: float
    mean_f1: float
    config: AdapterConfig
    n_samples: int
    n_positive: int
    n_features: int
    selected_features: Optional[int] = None
    feature_importance: Optional[np.ndarray] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        """Get a formatted summary of results."""
        lines = [
            f"Adapter Training Results",
            f"========================",
            f"Model type: {self.config.model_type}",
            f"Feature selection: {self.config.feature_selection}",
            f"Samples: {self.n_samples} ({self.n_positive} positive, {self.n_samples - self.n_positive} negative)",
            f"Features: {self.n_features}" + (f" -> {self.selected_features}" if self.selected_features else ""),
            f"",
            f"Cross-Validation Results ({len(self.fold_results)} folds):",
            f"  AUC:      {self.mean_auc:.4f} +/- {self.std_auc:.4f}",
            f"  Accuracy: {self.mean_accuracy:.4f}",
            f"  F1:       {self.mean_f1:.4f}",
        ]
        return "\n".join(lines)


class AdapterTrainer:
    """
    Training pipeline for adapter models.

    Handles feature extraction, cross-validation, and model training.
    """

    def __init__(
        self,
        model_type: str = "logistic",
        n_folds: int = 5,
        feature_selection: str = "none",
        n_features: int = 1000,
        api_url: Optional[str] = None,
        cache_path: Optional[str] = None,
        random_state: int = 42,
        # Semantic selection config
        target_property: Optional[str] = None,
        target_description: Optional[str] = None,
        semantic_model: str = "gemini-2.0-flash",
        semantic_max_categories: int = 10,
        semantic_cache_path: Optional[str] = None,
        # Vertex AI config
        use_vertex_ai: bool = False,
        vertex_project: Optional[str] = None,
        vertex_location: str = "us-central1",
    ):
        """
        Initialize the trainer.

        Args:
            model_type: "logistic" or "xgboost".
            n_folds: Number of cross-validation folds.
            feature_selection: "none", "variance", "topk", "category", or "semantic".
            n_features: Number of features for topk selection.
            api_url: ToxTransformer API URL.
            cache_path: Path to feature cache.
            random_state: Random seed for reproducibility.
            target_property: Target property name for semantic selection.
            target_description: Optional description for semantic selection.
            semantic_model: LLM model for semantic selection.
            semantic_max_categories: Max categories for semantic selection.
            semantic_cache_path: Cache path for semantic selection results.
            use_vertex_ai: Use Vertex AI instead of Google AI API.
            vertex_project: GCP project ID for Vertex AI.
            vertex_location: GCP region for Vertex AI.
        """
        self.model_type = model_type
        self.n_folds = n_folds
        self.feature_selection = feature_selection
        self.n_features = n_features
        self.random_state = random_state

        # Semantic selection config
        self.target_property = target_property
        self.target_description = target_description
        self.semantic_model = semantic_model
        self.semantic_max_categories = semantic_max_categories
        self.semantic_cache_path = semantic_cache_path or "cache/semantic_selections"

        # Vertex AI config
        self.use_vertex_ai = use_vertex_ai
        self.vertex_project = vertex_project
        self.vertex_location = vertex_location

        self.extractor = ToxTransformerFeatureExtractor(
            api_url=api_url,
            cache_path=cache_path,
        )

    def train(
        self,
        inchis: List[str],
        labels: List[int],
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> AdapterResult:
        """
        Train adapter with cross-validation.

        Args:
            inchis: List of InChI strings.
            labels: List of binary labels (0 or 1).
            model_kwargs: Additional kwargs for the model.

        Returns:
            AdapterResult with trained model and metrics.
        """
        model_kwargs = model_kwargs or {}

        # Convert to arrays
        inchis = list(inchis)
        y = np.array(labels, dtype=np.int32)

        logging.info(f"Training adapter on {len(inchis)} samples")
        logging.info(f"Class distribution: {np.sum(y == 0)} negative, {np.sum(y == 1)} positive")

        # Extract features
        logging.info("Extracting ToxTransformer features...")
        X = self.extractor.batch_extract(inchis)
        logging.info(f"Feature matrix shape: {X.shape}")

        # Create config
        config = AdapterConfig(
            model_type=self.model_type,
            feature_selection=self.feature_selection,
            n_features=self.n_features,
            random_state=self.random_state,
            target_property=self.target_property,
            target_description=self.target_description,
            semantic_model=self.semantic_model,
            semantic_max_categories=self.semantic_max_categories,
            semantic_cache_path=self.semantic_cache_path,
            use_vertex_ai=self.use_vertex_ai,
            vertex_project=self.vertex_project,
            vertex_location=self.vertex_location,
        )

        # Cross-validation
        logging.info(f"Running {self.n_folds}-fold cross-validation...")
        fold_results = self._cross_validate(X, y, config, model_kwargs)

        # Train final model on all data
        logging.info("Training final model on all data...")
        final_model = create_adapter(self.model_type, config, **model_kwargs)
        final_model.fit(X, y)

        # Compute summary stats
        aucs = [f.auc for f in fold_results]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_accuracy = np.mean([f.accuracy for f in fold_results])
        mean_f1 = np.mean([f.f1 for f in fold_results])

        # Get feature importance
        feature_importance = final_model.get_feature_importance()

        result = AdapterResult(
            model=final_model,
            fold_results=fold_results,
            mean_auc=mean_auc,
            std_auc=std_auc,
            mean_accuracy=mean_accuracy,
            mean_f1=mean_f1,
            config=config,
            n_samples=len(y),
            n_positive=int(np.sum(y == 1)),
            n_features=X.shape[1],
            selected_features=(
                len(final_model.selected_feature_indices)
                if final_model.selected_feature_indices is not None
                else None
            ),
            feature_importance=feature_importance,
        )

        logging.info(f"Training complete. AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

        return result

    def train_from_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_kwargs: Optional[Dict[str, Any]] = None,
    ) -> AdapterResult:
        """
        Train adapter from pre-extracted features.

        Args:
            X: Feature matrix of shape (n_samples, n_features).
            y: Labels of shape (n_samples,).
            model_kwargs: Additional kwargs for the model.

        Returns:
            AdapterResult with trained model and metrics.
        """
        model_kwargs = model_kwargs or {}
        y = np.array(y, dtype=np.int32)

        logging.info(f"Training adapter on {len(y)} samples with pre-extracted features")
        logging.info(f"Class distribution: {np.sum(y == 0)} negative, {np.sum(y == 1)} positive")

        # Create config
        config = AdapterConfig(
            model_type=self.model_type,
            feature_selection=self.feature_selection,
            n_features=self.n_features,
            random_state=self.random_state,
            target_property=self.target_property,
            target_description=self.target_description,
            semantic_model=self.semantic_model,
            semantic_max_categories=self.semantic_max_categories,
            semantic_cache_path=self.semantic_cache_path,
            use_vertex_ai=self.use_vertex_ai,
            vertex_project=self.vertex_project,
            vertex_location=self.vertex_location,
        )

        # Cross-validation
        logging.info(f"Running {self.n_folds}-fold cross-validation...")
        fold_results = self._cross_validate(X, y, config, model_kwargs)

        # Train final model on all data
        logging.info("Training final model on all data...")
        final_model = create_adapter(self.model_type, config, **model_kwargs)
        final_model.fit(X, y)

        # Compute summary stats
        aucs = [f.auc for f in fold_results]
        mean_auc = np.mean(aucs)
        std_auc = np.std(aucs)
        mean_accuracy = np.mean([f.accuracy for f in fold_results])
        mean_f1 = np.mean([f.f1 for f in fold_results])

        # Get feature importance
        feature_importance = final_model.get_feature_importance()

        result = AdapterResult(
            model=final_model,
            fold_results=fold_results,
            mean_auc=mean_auc,
            std_auc=std_auc,
            mean_accuracy=mean_accuracy,
            mean_f1=mean_f1,
            config=config,
            n_samples=len(y),
            n_positive=int(np.sum(y == 1)),
            n_features=X.shape[1],
            selected_features=(
                len(final_model.selected_feature_indices)
                if final_model.selected_feature_indices is not None
                else None
            ),
            feature_importance=feature_importance,
        )

        logging.info(f"Training complete. AUC: {mean_auc:.4f} +/- {std_auc:.4f}")

        return result

    def _cross_validate(
        self,
        X: np.ndarray,
        y: np.ndarray,
        config: AdapterConfig,
        model_kwargs: Dict[str, Any],
    ) -> List[FoldResult]:
        """Run k-fold cross-validation."""
        kfold = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        fold_results = []

        for fold, (train_idx, val_idx) in enumerate(kfold.split(X, y)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]

            # Train model
            model = create_adapter(self.model_type, config, **model_kwargs)
            model.fit(X_train, y_train)

            # Predict
            y_pred = model.predict(X_val)
            y_proba = model.predict_proba(X_val)[:, 1]

            # Compute metrics
            auc = roc_auc_score(y_val, y_proba)
            accuracy = accuracy_score(y_val, y_pred)
            precision = precision_score(y_val, y_pred, zero_division=0)
            recall = recall_score(y_val, y_pred, zero_division=0)
            f1 = f1_score(y_val, y_pred, zero_division=0)

            fold_result = FoldResult(
                fold=fold,
                auc=auc,
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1=f1,
                n_train=len(train_idx),
                n_val=len(val_idx),
                n_pos_train=int(np.sum(y_train)),
                n_pos_val=int(np.sum(y_val)),
            )

            fold_results.append(fold_result)
            logging.info(
                f"  Fold {fold + 1}/{self.n_folds}: AUC={auc:.4f}, Acc={accuracy:.4f}, F1={f1:.4f}"
            )

        return fold_results

    def save_result(self, result: AdapterResult, path: Union[str, Path]):
        """Save training result including model and metrics."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Save model
        result.model.save(path)

        # Save metrics separately as JSON
        import json

        metrics_path = path.with_suffix(".json")
        metrics = {
            "mean_auc": result.mean_auc,
            "std_auc": result.std_auc,
            "mean_accuracy": result.mean_accuracy,
            "mean_f1": result.mean_f1,
            "n_samples": result.n_samples,
            "n_positive": result.n_positive,
            "n_features": result.n_features,
            "selected_features": result.selected_features,
            "model_type": result.config.model_type,
            "feature_selection": result.config.feature_selection,
            "fold_results": [
                {
                    "fold": f.fold,
                    "auc": f.auc,
                    "accuracy": f.accuracy,
                    "precision": f.precision,
                    "recall": f.recall,
                    "f1": f.f1,
                    "n_train": f.n_train,
                    "n_val": f.n_val,
                    "n_pos_train": f.n_pos_train,
                    "n_pos_val": f.n_pos_val,
                }
                for f in result.fold_results
            ],
        }

        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)

        logging.info(f"Saved model to {path}")
        logging.info(f"Saved metrics to {metrics_path}")
