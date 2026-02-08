"""
Adapter Evaluator

Evaluation utilities for adapter models including baseline comparisons.
"""

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from adapter.feature_extractor import ToxTransformerFeatureExtractor
from adapter.models import BaseAdapter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
)


@dataclass
class EvaluationResult:
    """Results from model evaluation."""
    auc: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    n_samples: int
    n_positive: int
    confusion_matrix: np.ndarray
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray
    fpr: Optional[np.ndarray] = None
    tpr: Optional[np.ndarray] = None
    thresholds: Optional[np.ndarray] = None

    def summary(self) -> str:
        """Get formatted summary of evaluation."""
        lines = [
            f"Evaluation Results",
            f"==================",
            f"Samples: {self.n_samples} ({self.n_positive} positive)",
            f"AUC:       {self.auc:.4f}",
            f"Accuracy:  {self.accuracy:.4f}",
            f"Precision: {self.precision:.4f}",
            f"Recall:    {self.recall:.4f}",
            f"F1:        {self.f1:.4f}",
            f"",
            f"Confusion Matrix:",
            f"  TN={self.confusion_matrix[0, 0]}, FP={self.confusion_matrix[0, 1]}",
            f"  FN={self.confusion_matrix[1, 0]}, TP={self.confusion_matrix[1, 1]}",
        ]
        return "\n".join(lines)


@dataclass
class BaselineResult:
    """Results from baseline comparison."""
    random_auc: float
    majority_accuracy: float
    majority_class: int
    stratified_auc: float


class AdapterEvaluator:
    """
    Evaluator for adapter models.

    Provides evaluation on holdout sets and baseline comparisons.
    """

    def __init__(
        self,
        api_url: Optional[str] = None,
        cache_path: Optional[str] = None,
    ):
        """
        Initialize the evaluator.

        Args:
            api_url: ToxTransformer API URL.
            cache_path: Path to feature cache.
        """
        self.extractor = ToxTransformerFeatureExtractor(
            api_url=api_url,
            cache_path=cache_path,
        )

    def evaluate(
        self,
        model: BaseAdapter,
        inchis: List[str],
        labels: List[int],
        compute_curves: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate adapter on a test set.

        Args:
            model: Trained adapter model.
            inchis: Test set InChI strings.
            labels: Test set labels.
            compute_curves: Whether to compute ROC curve data.

        Returns:
            EvaluationResult with all metrics.
        """
        y_true = np.array(labels, dtype=np.int32)

        # Extract features
        logging.info(f"Extracting features for {len(inchis)} test samples...")
        X = self.extractor.batch_extract(inchis)

        return self.evaluate_from_features(model, X, y_true, compute_curves)

    def evaluate_from_features(
        self,
        model: BaseAdapter,
        X: np.ndarray,
        y_true: np.ndarray,
        compute_curves: bool = True,
    ) -> EvaluationResult:
        """
        Evaluate adapter from pre-extracted features.

        Args:
            model: Trained adapter model.
            X: Feature matrix.
            y_true: True labels.
            compute_curves: Whether to compute ROC curve data.

        Returns:
            EvaluationResult with all metrics.
        """
        # Predict
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        # Compute metrics
        auc = roc_auc_score(y_true, y_proba)
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        # ROC curve
        fpr, tpr, thresholds = None, None, None
        if compute_curves:
            fpr, tpr, thresholds = roc_curve(y_true, y_proba)

        return EvaluationResult(
            auc=auc,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1=f1,
            n_samples=len(y_true),
            n_positive=int(np.sum(y_true)),
            confusion_matrix=cm,
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
            fpr=fpr,
            tpr=tpr,
            thresholds=thresholds,
        )

    def compute_baselines(self, y_true: np.ndarray, n_random: int = 1000) -> BaselineResult:
        """
        Compute baseline metrics for comparison.

        Args:
            y_true: True labels.
            n_random: Number of random samples for random baseline.

        Returns:
            BaselineResult with baseline metrics.
        """
        y_true = np.array(y_true, dtype=np.int32)

        # Random baseline - average AUC of random predictions
        random_aucs = []
        for _ in range(n_random):
            y_random = np.random.rand(len(y_true))
            try:
                random_aucs.append(roc_auc_score(y_true, y_random))
            except ValueError:
                continue
        random_auc = np.mean(random_aucs) if random_aucs else 0.5

        # Majority class baseline
        n_pos = np.sum(y_true)
        n_neg = len(y_true) - n_pos
        majority_class = 1 if n_pos > n_neg else 0
        majority_accuracy = max(n_pos, n_neg) / len(y_true)

        # Stratified random baseline
        stratified_aucs = []
        p_pos = n_pos / len(y_true)
        for _ in range(n_random):
            y_stratified = np.random.rand(len(y_true))
            # Adjust to have roughly correct class proportions
            threshold = np.percentile(y_stratified, 100 * (1 - p_pos))
            y_stratified = (y_stratified > threshold).astype(int)
            try:
                stratified_aucs.append(roc_auc_score(y_true, y_stratified))
            except ValueError:
                continue
        stratified_auc = np.mean(stratified_aucs) if stratified_aucs else 0.5

        return BaselineResult(
            random_auc=random_auc,
            majority_accuracy=majority_accuracy,
            majority_class=majority_class,
            stratified_auc=stratified_auc,
        )

    def compare_to_baseline(
        self,
        model: BaseAdapter,
        X: np.ndarray,
        y_true: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Evaluate model and compare to baselines.

        Args:
            model: Trained adapter model.
            X: Feature matrix.
            y_true: True labels.

        Returns:
            Dict with model metrics and baseline comparisons.
        """
        # Evaluate model
        eval_result = self.evaluate_from_features(model, X, y_true)

        # Compute baselines
        baselines = self.compute_baselines(y_true)

        return {
            "model_auc": eval_result.auc,
            "model_accuracy": eval_result.accuracy,
            "model_f1": eval_result.f1,
            "random_auc": baselines.random_auc,
            "majority_accuracy": baselines.majority_accuracy,
            "auc_improvement": eval_result.auc - baselines.random_auc,
            "accuracy_improvement": eval_result.accuracy - baselines.majority_accuracy,
        }

    def find_optimal_threshold(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        metric: str = "f1",
    ) -> Tuple[float, float]:
        """
        Find optimal classification threshold.

        Args:
            y_true: True labels.
            y_proba: Predicted probabilities.
            metric: Metric to optimize ("f1", "precision", "recall", "balanced").

        Returns:
            Tuple of (optimal_threshold, best_metric_value).
        """
        precision_arr, recall_arr, thresholds = precision_recall_curve(y_true, y_proba)

        # Remove last element (which has threshold=1)
        precision_arr = precision_arr[:-1]
        recall_arr = recall_arr[:-1]

        if metric == "f1":
            with np.errstate(divide="ignore", invalid="ignore"):
                f1_arr = 2 * (precision_arr * recall_arr) / (precision_arr + recall_arr)
                f1_arr = np.nan_to_num(f1_arr)
            best_idx = np.argmax(f1_arr)
            return thresholds[best_idx], f1_arr[best_idx]

        elif metric == "precision":
            best_idx = np.argmax(precision_arr)
            return thresholds[best_idx], precision_arr[best_idx]

        elif metric == "recall":
            best_idx = np.argmax(recall_arr)
            return thresholds[best_idx], recall_arr[best_idx]

        elif metric == "balanced":
            # Youden's J statistic
            fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)
            j_stat = tpr - fpr
            best_idx = np.argmax(j_stat)
            return roc_thresholds[best_idx], j_stat[best_idx]

        else:
            raise ValueError(f"Unknown metric: {metric}")


def format_comparison_table(
    results: Dict[str, EvaluationResult],
    baselines: Optional[BaselineResult] = None,
) -> str:
    """
    Format comparison table for multiple models.

    Args:
        results: Dict mapping model names to EvaluationResult.
        baselines: Optional baseline results.

    Returns:
        Formatted table string.
    """
    lines = []
    lines.append(f"{'Model':<20} {'AUC':>8} {'Acc':>8} {'Prec':>8} {'Recall':>8} {'F1':>8}")
    lines.append("-" * 60)

    for name, result in results.items():
        lines.append(
            f"{name:<20} {result.auc:>8.4f} {result.accuracy:>8.4f} "
            f"{result.precision:>8.4f} {result.recall:>8.4f} {result.f1:>8.4f}"
        )

    if baselines:
        lines.append("-" * 60)
        lines.append(f"{'Random':<20} {baselines.random_auc:>8.4f}")
        lines.append(f"{'Majority':<20} {'':>8} {baselines.majority_accuracy:>8.4f}")

    return "\n".join(lines)
