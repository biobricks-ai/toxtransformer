"""
Evaluation system for model training.

This module provides an abstract evaluation framework with concrete implementations:
- Evaluator: Abstract base class defining the evaluation interface
- DefaultEvaluator: Standard evaluation with overall metrics
- StratifiedEvaluator: Per-property evaluation with detailed breakdowns
- StratifiedGroupEvaluator: Per-group evaluation with properties grouped by data source
- InverseFrequencyWeightedEvaluator: Weighted evaluation using inverse task frequencies
"""

from .evaluator import Evaluator
from .default_evaluator import DefaultEvaluator
from .stratified_evaluator import StratifiedEvaluator
from .stratified_group_evaluator import StratifiedGroupEvaluator
from .invfreq_evaluator import InverseFrequencyWeightedEvaluator

__all__ = [
    "Evaluator",
    "DefaultEvaluator", 
    "StratifiedEvaluator",
    "StratifiedGroupEvaluator",
    "InverseFrequencyWeightedEvaluator"
]
