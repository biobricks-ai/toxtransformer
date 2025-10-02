import logging
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast

from .evaluator import Evaluator


class DefaultEvaluator(Evaluator):
    """
    Default evaluator that computes overall loss and classification metrics.
    
    This replicates the evaluation behavior from the original trainer_core.py.
    """
    
    def evaluate(self, model: nn.Module, valdl, max_eval_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate the model on validation data with standard metrics.
        
        Args:
            model: The model to evaluate
            valdl: Validation dataloader
            max_eval_batches: Maximum number of batches to evaluate
            
        Returns:
            Dictionary containing loss, auc, and bac metrics
        """
        torch.cuda.empty_cache()
        max_eval_batches = max_eval_batches or len(valdl)
        max_eval_batches = min(max_eval_batches, len(valdl))

        model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds, all_targets = [], []

        if dist.is_initialized():
            dist.barrier(device_ids=[self.rank])

        for i, (selfies, properties, values, mask) in enumerate(valdl):
            if i >= max_eval_batches:
                break

            selfies = selfies.to(self.rank, non_blocking=True)
            properties = properties.to(self.rank, non_blocking=True)
            values = values.to(self.rank, non_blocking=True)
            mask = mask.to(self.rank, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(selfies, properties, values, mask)

                # Compute masked cross-entropy loss
                B, T, C = logits.shape
                logits_f = logits.reshape(-1, C)
                vals_f = values.reshape(-1)
                mask_f = mask.reshape(-1)
                
                if mask_f.any():
                    loss = nn.functional.cross_entropy(logits_f[mask_f], vals_f[mask_f])
                else:
                    loss = logits_f.new_zeros(())

                # Collect predictions and targets for metrics
                probs = F.softmax(logits, dim=-1)
                probs_f = probs.reshape(-1, C)

                if mask_f.any():
                    all_preds.append(probs_f[mask_f])
                    all_targets.append(vals_f[mask_f])

                total_loss += float(loss)
                num_batches += 1

        # Gather loss across ranks
        if dist.is_initialized():
            loss_t = torch.tensor(total_loss, device=self.rank)
            nb_t = torch.tensor(num_batches, device=self.rank)
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(nb_t, op=dist.ReduceOp.SUM)
            mean_loss = loss_t.item() / max(1, nb_t.item())
        else:
            mean_loss = total_loss / max(1, num_batches)

        # Compute metrics
        predictions, targets = self._gather_predictions_and_targets(all_preds, all_targets)
        
        if self.rank == 0:
            logging.info(f"Metrics computed on {len(targets)} samples, targets: {np.unique(targets)}")
            
        metrics = self._compute_metrics(predictions, targets)
        
        # Combine loss with classification metrics
        result = {"loss": mean_loss}
        result.update(metrics)
        
        if self.rank == 0:
            logging.info(f"📊 EVAL METRICS - Loss: {result['loss']:.4f}, AUC: {result['auc']:.4f}, BAC: {result['bac']:.4f}")
        
        return result
