# helper/trainer/__init__.py
"""
Training utilities and evaluators for the ToxTransformer project
"""

from .trainer_core import TrainerCore
from .evaluator import StratifiedEvaluator, StratifiedGroupEvaluator

# Import existing trainers if available
from .invfreq_trainer import InverseFrequencyWeightedTrainer

from .selfies_properties_values_trainer import SelfiesPropertiesValuesTrainer

from .curriculum_trainer import AccumulatedStratifiedPropertyWeightedTrainer

__all__ = [
    'TrainerCore',
    'StratifiedEvaluator',
    'StratifiedGroupEvaluator',
    'InverseFrequencyWeightedTrainer',
    'SelfiesPropertiesValuesTrainer',
    'AccumulatedStratifiedPropertyWeightedTrainer',
]
