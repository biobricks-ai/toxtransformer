import logging
from abc import ABC, abstractmethod
from typing import Dict, Optional, Tuple, List
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from sklearn.metrics import roc_auc_score, balanced_accuracy_score


class Evaluator(ABC):
    """
    Abstract base class for model evaluation.
    
    Handles the evaluation loop and delegates specific evaluation logic to subclasses.
    """
    
    def __init__(self, rank: int):
        self.rank = rank
    
    @abstractmethod
    def evaluate(self, model: nn.Module, valdl, max_eval_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate the model on validation data.
        
        Args:
            model: The model to evaluate
            valdl: Validation dataloader
            max_eval_batches: Maximum number of batches to evaluate (None for all)
            
        Returns:
            Dictionary containing evaluation metrics
        """
        pass
    
    def _gather_predictions_and_targets(self, all_preds: List[torch.Tensor], 
                                      all_targets: List[torch.Tensor]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Gather predictions and targets across all ranks if using distributed training.
        
        Args:
            all_preds: List of prediction tensors
            all_targets: List of target tensors
            
        Returns:
            Tuple of (predictions, targets) as numpy arrays
        """
        if not all_preds:
            return np.array([]).reshape(0, 2), np.array([])
            
        ap = torch.cat(all_preds, dim=0)
        at = torch.cat(all_targets, dim=0)

        if dist.is_initialized():
            local_len = torch.tensor([ap.size(0)], device=self.rank)
            lengths = [torch.zeros_like(local_len) for _ in range(dist.get_world_size())]
            dist.all_gather(lengths, local_len)
            
            lengths_list = [l.item() for l in lengths]
            max_len = max(lengths_list)
            
            if max_len > ap.size(0):
                ap = F.pad(ap, (0, 0, 0, max_len - ap.size(0)))
                at = F.pad(at, (0, max_len - at.size(0)), value=-1)

            g_preds = [torch.zeros_like(ap) for _ in range(dist.get_world_size())]
            g_targs = [torch.zeros_like(at) for _ in range(dist.get_world_size())]
            dist.all_gather(g_preds, ap)
            dist.all_gather(g_targs, at)

            # Only take valid samples from each rank
            ap_list = [g_preds[i][:lengths_list[i]] for i in range(len(lengths_list)) if lengths_list[i] > 0]
            at_list = [g_targs[i][:lengths_list[i]] for i in range(len(lengths_list)) if lengths_list[i] > 0]
            
            ap = torch.cat(ap_list, dim=0).float().cpu().numpy() if ap_list else np.array([]).reshape(0, ap.size(1))
            at = torch.cat(at_list, dim=0).cpu().numpy() if at_list else np.array([])
        else:
            ap = ap.float().cpu().numpy()
            at = at.cpu().numpy()

        # Filter out invalid targets
        valid = at >= 0
        ap = ap[valid]
        at = at[valid]
        
        return ap, at
    
    def _compute_metrics(self, predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        """
        Compute standard classification metrics.
        
        Args:
            predictions: Prediction probabilities [N, num_classes]
            targets: Target labels [N]
            
        Returns:
            Dictionary containing computed metrics
        """
        if len(targets) == 0:
            return {"auc": 0.0, "bac": 0.0}
            
        unique_targets = np.unique(targets)
        
        if len(unique_targets) == 2 and predictions.shape[1] >= 2:
            try:
                auc = roc_auc_score(targets, predictions[:, 1])
                bac = balanced_accuracy_score(targets, predictions.argmax(axis=1))
            except Exception as e:
                if self.rank == 0:
                    logging.warning(f"Metric computation error: {e}")
                auc = 0.0
                bac = 0.0
        else:
            auc, bac = 0.0, 0.0
            
        return {"auc": auc, "bac": bac}
