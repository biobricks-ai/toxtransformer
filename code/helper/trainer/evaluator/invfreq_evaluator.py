import logging
from typing import Dict, Optional, List, Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

from .evaluator import Evaluator


class InverseFrequencyWeightedEvaluator(Evaluator):
    """
    Evaluator that uses inverse frequency weighting to compute weighted metrics.
    
    This evaluator mirrors the weighting scheme used in InverseFrequencyWeightedTrainer
    to provide consistent evaluation metrics that reflect the same task balancing.
    """
    
    def __init__(self, 
                 rank: int, 
                 tokenizer=None,
                 max_weight: float = 10.0,
                 ema_eps: float = 1e-6):
        """
        Initialize the inverse frequency weighted evaluator.
        
        Args:
            rank: Device rank for distributed training
            tokenizer: Tokenizer with assay_indexes() method
            max_weight: Maximum weight to clamp to (should match trainer)
            ema_eps: Small constant for numerical stability (should match trainer)
        """
        super().__init__(rank, tokenizer)
        self.max_weight = max_weight
        self.ema_eps = ema_eps
        
        # Build stable task mapping (same as trainer)
        if tokenizer is not None:
            assay_tokens = sorted(self.tokenizer.assay_indexes().values())
            self._minprop = assay_tokens[0]
            self._num_tasks = len(assay_tokens)
            self._contiguous = (assay_tokens == list(range(self._minprop, self._minprop + self._num_tasks)))
            
            if not self._contiguous:
                self._prop_to_index = {p: i for i, p in enumerate(assay_tokens)}
            else:
                self._prop_to_index = None
        else:
            self._minprop = 0
            self._num_tasks = 0
            self._contiguous = True
            self._prop_to_index = None
    
    def _prop_to_slot(self, props_flat_valid: torch.Tensor) -> torch.Tensor:
        """
        Map raw property tokens -> stable slot [0..num_tasks-1].
        Same logic as in InverseFrequencyWeightedTrainer.
        """
        if self._contiguous:
            return (props_flat_valid - self._minprop).clamp_min(0)
        
        # explicit map
        mapped = [self._prop_to_index.get(int(p), -1) for p in props_flat_valid.tolist()]
        out = torch.tensor(mapped, device=props_flat_valid.device, dtype=torch.long)
        return out
    
    def _compute_task_frequencies(self, valdl, max_eval_batches: Optional[int] = None) -> torch.Tensor:
        """
        Compute task frequencies from validation data to derive weights.
        
        Args:
            valdl: Validation dataloader
            max_eval_batches: Maximum batches to use for frequency estimation
            
        Returns:
            Tensor of task frequencies [num_tasks]
        """
        if self._num_tasks == 0:
            return torch.ones(1, device=self.rank)
            
        max_eval_batches = max_eval_batches or len(valdl)
        max_eval_batches = min(max_eval_batches, len(valdl))
        
        total_counts = torch.zeros(self._num_tasks, device=self.rank, dtype=torch.long)
        total_positions = 0
        
        for i, (_, properties, _, mask) in enumerate(valdl):
            if i >= max_eval_batches:
                break
                
            properties = properties.to(self.rank, non_blocking=True)
            mask = mask.to(self.rank, non_blocking=True)
            
            mask_f = mask.view(-1)
            if mask_f.any():
                props_f = properties.view(-1)
                pos = torch.nonzero(mask_f, as_tuple=False).squeeze(1)
                props_valid = props_f[pos]
                
                slots = self._prop_to_slot(props_valid)
                valid_mask = (slots >= 0) & (slots < self._num_tasks)
                slots = slots[valid_mask]
                
                # Ensure slots are on the correct device
                slots = slots.to(self.rank)
                
                counts = torch.bincount(slots, minlength=self._num_tasks)
                total_counts += counts
                total_positions += pos.numel()
        
        # Gather across ranks if distributed
        if dist.is_initialized():
            dist.all_reduce(total_counts, op=dist.ReduceOp.SUM)
            total_pos_t = torch.tensor(total_positions, device=self.rank)
            dist.all_reduce(total_pos_t, op=dist.ReduceOp.SUM)
            total_positions = total_pos_t.item()
        
        # Convert to frequencies
        if total_positions > 0:
            freqs = total_counts.float() / total_positions
        else:
            freqs = torch.full((self._num_tasks,), 1.0 / max(1, self._num_tasks), device=self.rank)
            
        return freqs
    
    def _get_inverse_weights(self, frequencies: torch.Tensor) -> torch.Tensor:
        """
        Compute inverse frequency weights, same logic as trainer.
        
        Args:
            frequencies: Task frequencies [num_tasks]
            
        Returns:
            Normalized and clamped inverse weights [num_tasks]
        """
        weights = 1.0 / (self.ema_eps + frequencies)
        # Normalize first
        weights = weights / weights.mean().clamp_min(1e-8)
        # Then clamp
        if self.max_weight is not None:
            weights = torch.clamp(weights, max=self.max_weight)
        return weights
    
    def _compute_weighted_metrics_per_task(self, 
                                         predictions: np.ndarray, 
                                         targets: np.ndarray, 
                                         task_ids: np.ndarray,
                                         weights: torch.Tensor) -> Dict[str, float]:
        """
        Compute weighted metrics across tasks.
        
        Args:
            predictions: Prediction probabilities [N, num_classes]
            targets: Target labels [N]
            task_ids: Task IDs for each sample [N]
            weights: Task weights [num_tasks]
            
        Returns:
            Dictionary with weighted metrics
        """
        if len(targets) == 0:
            return {"weighted_auc": 0.0, "weighted_bac": 0.0}
        
        weights_np = weights.cpu().numpy()
        unique_tasks = np.unique(task_ids)
        
        task_aucs = []
        task_bacs = []
        task_weights = []
        
        for task_id in unique_tasks:
            if task_id < 0 or task_id >= len(weights_np):
                continue
                
            mask = task_ids == task_id
            task_preds = predictions[mask]
            task_targets = targets[mask]
            
            if len(task_targets) == 0:
                continue
                
            unique_targets = np.unique(task_targets)
            
            # Compute metrics if we have binary classification
            if len(unique_targets) == 2 and task_preds.shape[1] >= 2:
                try:
                    auc = roc_auc_score(task_targets, task_preds[:, 1])
                    bac = balanced_accuracy_score(task_targets, task_preds.argmax(axis=1))
                    
                    task_aucs.append(auc)
                    task_bacs.append(bac)
                    task_weights.append(weights_np[task_id])
                except Exception as e:
                    if self.rank == 0:
                        logging.warning(f"Task {task_id} metric error: {e}")
                    continue
        
        if not task_aucs:
            return {"weighted_auc": 0.0, "weighted_bac": 0.0}
        
        # Compute weighted averages
        task_aucs = np.array(task_aucs)
        task_bacs = np.array(task_bacs)
        task_weights = np.array(task_weights)
        
        # Normalize weights for this subset of tasks
        task_weights = task_weights / task_weights.sum()
        
        weighted_auc = np.sum(task_aucs * task_weights)
        weighted_bac = np.sum(task_bacs * task_weights)
        
        return {
            "weighted_auc": weighted_auc,
            "weighted_bac": weighted_bac,
            "num_tasks_evaluated": len(task_aucs)
        }
    
    def evaluate(self, model: nn.Module, valdl, max_eval_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate the model with inverse frequency weighting.
        
        Args:
            model: The model to evaluate
            valdl: Validation dataloader
            max_eval_batches: Maximum number of batches to evaluate
            
        Returns:
            Dictionary containing weighted loss and metrics
        """
        torch.cuda.empty_cache()
        max_eval_batches = max_eval_batches or len(valdl)
        max_eval_batches = min(max_eval_batches, len(valdl))

        model.eval()
        
        # First pass: compute task frequencies
        if self.rank == 0:
            logging.info("Computing task frequencies for weighted evaluation...")
        
        frequencies = self._compute_task_frequencies(valdl, max_eval_batches)
        weights = self._get_inverse_weights(frequencies)
        
        if self.rank == 0:
            freqs_np = frequencies.cpu().numpy()
            weights_np = weights.cpu().numpy()
            logging.info(f"Task frequencies: min={freqs_np.min():.6f}, max={freqs_np.max():.6f}")
            logging.info(f"Task weights: min={weights_np.min():.3f}, max={weights_np.max():.3f}")

        # Second pass: evaluate with weighting
        total_weighted_loss = 0.0
        total_weight = 0.0
        all_preds, all_targets, all_task_ids = [], [], []

        if dist.is_initialized():
            dist.barrier(device_ids=[self.rank])

        lossfn = nn.CrossEntropyLoss(reduction='none')
        
        for i, (selfies, properties, values, mask) in enumerate(valdl):
            if i >= max_eval_batches:
                break

            selfies = selfies.to(self.rank, non_blocking=True)
            properties = properties.to(self.rank, non_blocking=True)
            values = values.to(self.rank, non_blocking=True)
            mask = mask.to(self.rank, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(selfies, properties, values, mask)

                B, T, C = logits.shape
                logits_f = logits.reshape(-1, C)
                vals_f = values.reshape(-1)
                props_f = properties.reshape(-1)
                mask_f = mask.reshape(-1)
                
                if not mask_f.any():
                    continue
                
                # Get valid positions
                pos = torch.nonzero(mask_f, as_tuple=False).squeeze(1)
                props_valid = props_f[pos]
                slots = self._prop_to_slot(props_valid)
                
                valid_mask = (slots >= 0) & (slots < self._num_tasks)
                slots = slots[valid_mask]
                pos = pos[valid_mask]
                
                if len(pos) == 0:
                    continue
                
                # Ensure pos tensor is on the same device as logits
                pos = pos.to(logits_f.device)
                slots = slots.to(self.rank)
                
                # Compute per-sample losses
                sample_losses = lossfn(logits_f[pos], vals_f[pos])
                
                # Apply task weights
                task_weights_batch = weights[slots]
                weighted_losses = sample_losses * task_weights_batch
                
                batch_weighted_loss = weighted_losses.sum()
                batch_weight = task_weights_batch.sum()
                
                total_weighted_loss += float(batch_weighted_loss)
                total_weight += float(batch_weight)
                
                # Collect predictions and targets with task IDs
                probs = F.softmax(logits, dim=-1)
                probs_f = probs.reshape(-1, C)
                
                all_preds.append(probs_f[pos])
                all_targets.append(vals_f[pos])
                all_task_ids.append(slots)

        # Gather loss across ranks
        if dist.is_initialized():
            loss_t = torch.tensor(total_weighted_loss, device=self.rank)
            weight_t = torch.tensor(total_weight, device=self.rank)
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(weight_t, op=dist.ReduceOp.SUM)
            mean_weighted_loss = loss_t.item() / max(1e-8, weight_t.item())
        else:
            mean_weighted_loss = total_weighted_loss / max(1e-8, total_weight)

        # Gather predictions and targets
        predictions, targets = self._gather_predictions_and_targets(all_preds, all_targets)
        
        # Gather task IDs
        if all_task_ids:
            task_ids_tensor = torch.cat(all_task_ids, dim=0)
            if dist.is_initialized():
                # Similar gathering logic for task IDs
                gathered_task_ids = [torch.zeros_like(task_ids_tensor) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_task_ids, task_ids_tensor)
                task_ids = torch.cat(gathered_task_ids, dim=0).cpu().numpy()
                # Trim to match predictions length
                task_ids = task_ids[:len(targets)]
            else:
                task_ids = task_ids_tensor.cpu().numpy()
        else:
            task_ids = np.array([])

        # Compute standard metrics
        standard_metrics = self._compute_metrics(predictions, targets)
        
        # Compute weighted metrics
        weighted_metrics = self._compute_weighted_metrics_per_task(
            predictions, targets, task_ids, weights
        )
        
        # Combine all metrics
        result = {
            "weighted_loss": mean_weighted_loss,
            "loss": standard_metrics.get("loss", mean_weighted_loss),  # fallback
            "auc": standard_metrics["auc"],
            "bac": standard_metrics["bac"],
        }
        result.update(weighted_metrics)
        
        if self.rank == 0:
            logging.info(f"📊 WEIGHTED EVAL METRICS:")
            logging.info(f"   Weighted Loss: {result['weighted_loss']:.4f}")
            logging.info(f"   Standard - AUC: {result['auc']:.4f}, BAC: {result['bac']:.4f}")
            logging.info(f"   Weighted - AUC: {result['weighted_auc']:.4f}, BAC: {result['weighted_bac']:.4f}")
            logging.info(f"   Tasks evaluated: {result.get('num_tasks_evaluated', 0)}")
        
        return result
