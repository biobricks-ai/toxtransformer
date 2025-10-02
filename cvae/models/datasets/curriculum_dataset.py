import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
from typing import Dict, List, Optional
import numpy as np


class CurriculumDataset(Dataset):
    """
    A curriculum learning wrapper that samples based on difficulty scores.
    
    Samples with higher scores (more difficult) are sampled more frequently.
    The sampling probability is proportional to score^temperature.
    """
    
    def __init__(self, base_dataset: Dataset, temperature: float = 2.0):
        """
        Args:
            base_dataset: The underlying dataset to wrap
            temperature: Controls sampling sharpness. Higher = more focus on difficult samples
        """
        self.base_dataset = base_dataset
        self.temperature = temperature
        
        # Initialize uniform scores
        self.scores = torch.ones(len(base_dataset), dtype=torch.float32)
        self._update_sampling_probs()
    
    def _update_sampling_probs(self):
        """Update sampling probabilities based on current scores."""
        # Use temperature to control how much we focus on difficult samples
        weighted_scores = self.scores ** self.temperature
        self.sampling_probs = weighted_scores / weighted_scores.sum()
        
        # Convert to numpy for faster sampling
        self.sampling_probs_np = self.sampling_probs.numpy()
    
    def update_scores(self, scores: torch.Tensor):
        """
        Update sample scores directly.
        
        Args:
            scores: Tensor of shape [len(dataset)] with new scores
        """
        assert len(scores) == len(self.base_dataset), f"Score length {len(scores)} != dataset length {len(self.base_dataset)}"
        self.scores = scores.clone().float()
        self._update_sampling_probs()
    
    def sample_indices(self, batch_size: int) -> torch.Tensor:
        """
        Sample indices based on difficulty scores.
        
        Returns:
            Tensor of sampled indices
        """
        indices = np.random.choice(
            len(self.base_dataset),
            size=batch_size,
            replace=True,
            p=self.sampling_probs_np
        )
        return torch.from_numpy(indices)
    
    def __len__(self):
        return len(self.base_dataset)
    
    def sample(self):
        """
        Sample a single item based on difficulty scores.
        
        Returns:
            The sampled item from the base dataset
        """
        idx = np.random.choice(len(self.base_dataset), p=self.sampling_probs_np)
        return self.base_dataset[idx]
    
    def __getitem__(self, idx):
        # Always sample based on scores, ignore the idx
        return self.sample()
    
    def get_score_stats(self) -> Dict[str, float]:
        """Get statistics about current scores for monitoring."""
        return {
            'mean_score': self.scores.mean().item(),
            'std_score': self.scores.std().item(),
            'min_score': self.scores.min().item(),
            'max_score': self.scores.max().item(),
        }


class CurriculumBatchSampler:
    """
    Custom batch sampler that uses curriculum dataset's scoring.
    Use this with DataLoader instead of default sampler.
    """
    
    def __init__(self, curriculum_dataset: CurriculumDataset, batch_size: int, num_batches: int):
        self.curriculum_dataset = curriculum_dataset
        self.batch_size = batch_size
        self.num_batches = num_batches
    
    def __iter__(self):
        for _ in range(self.num_batches):
            indices = self.curriculum_dataset.sample_indices(self.batch_size)
            yield indices.tolist()
    
    def __len__(self):
        return self.num_batches


# Example usage:
"""
# Wrap your existing dataset
from cvae.models.datasets import InMemorySelfiesPropertiesValuesDataset, CurriculumDataset
import itertools, uuid, pathlib
import pandas as pd, tqdm, sklearn.metrics, torch, numpy as np, os
import cvae.tokenizer, cvae.models.multitask_transformer as mt, cvae.models.mixture_experts as me
import logging
from cvae.tokenizer import SelfiesPropertyValTokenizer
from pyspark.sql.functions import col, when, countDistinct
from pyspark.sql import SparkSession
from pyspark.sql.functions import split, col, when
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, log_loss
from cvae.models.datasets import InMemorySelfiesPropertiesValuesDataset
from torch.nn import functional as F
from tqdm import tqdm


tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

trnds = InMemorySelfiesPropertiesValuesDataset("cache/build_tensordataset/multitask_tensors/tmp", tokenizer, nprops=5)
curriculum_dataset = CurriculumDataset(trnds, temperature=2.0)

# Sample individual items
sample = curriculum_dataset.sample()

# Or sample multiple items
samples = [curriculum_dataset.sample() for _ in range(batch_size)]

# Update scores directly (e.g., based on your computed difficulty)
new_scores = torch.randn(len(curriculum_dataset))  # Your computed scores
curriculum_dataset.update_scores(new_scores)

# Create custom batch sampler for DataLoader
batch_sampler = CurriculumBatchSampler(curriculum_dataset, batch_size=32, num_batches=100)
dataloader = DataLoader(curriculum_dataset, batch_sampler=batch_sampler)

# Monitor curriculum learning
stats = curriculum_dataset.get_score_stats()
print(f"Score stats: {stats}")
"""