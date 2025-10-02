import logging
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast

from .evaluator import Evaluator


class StratifiedEvaluator(Evaluator):
    """
    Simplified stratified evaluator that computes metrics separately for each property/task.
    
    Provides:
    - Stratified loss (mean of per-property losses)
    - Overall loss (weighted by sample count)
    - AUC and BAC metrics
    """

    def __init__(self, rank: int, num_tasks: int):
        super().__init__(rank)
        self.num_tasks = num_tasks

    def evaluate(self, model: nn.Module, valdl, max_eval_batches: Optional[int] = None) -> Dict[str, float]:
        """Evaluate the model with per-property metrics."""
        torch.cuda.empty_cache()
        max_eval_batches = min(max_eval_batches or len(valdl), len(valdl))

        model.eval()
        
        # Track metrics
        total_loss = 0.0
        total_samples = 0
        all_preds, all_targets = [], []
        
        # Per-property tracking - use GPU tensors for efficiency
        prop_losses = torch.zeros(self.num_tasks, device=self.rank)
        prop_counts = torch.zeros(self.num_tasks, device=self.rank)

        if dist.is_initialized():
            dist.barrier(device_ids=[self.rank])

        with torch.no_grad():
            for i, (selfies, properties, values, mask) in enumerate(valdl):
                if i >= max_eval_batches:
                    break
                
                if i % 10 == 0:  # Reduce logging frequency
                    logging.info(f"Evaluating batch {i}/{max_eval_batches}")

                # Move to device
                selfies = selfies.to(self.rank, non_blocking=True)
                properties = properties.to(self.rank, non_blocking=True)
                values = values.to(self.rank, non_blocking=True)
                mask = mask.to(self.rank, non_blocking=True)

                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits = model(selfies, properties, values, mask)

                    # Reshape for loss computation
                    B, T, C = logits.shape
                    logits_f = logits.view(-1, C)
                    vals_f = values.view(-1)
                    mask_f = mask.view(-1)
                    props_f = properties.view(-1)
                    
                    # Only compute on valid positions
                    if mask_f.any():
                        valid_logits = logits_f[mask_f]
                        valid_vals = vals_f[mask_f]
                        valid_props = props_f[mask_f]
                        
                        # Overall loss
                        batch_loss = F.cross_entropy(valid_logits, valid_vals, reduction='sum', label_smoothing=.05)
                        total_loss += batch_loss.item()
                        total_samples += mask_f.sum().item()
                        
                        # Per-property losses - vectorized computation
                        for task_id in torch.unique(valid_props):
                            if task_id < self.num_tasks:
                                task_mask = (valid_props == task_id)
                                if task_mask.any():
                                    task_loss = F.cross_entropy(
                                        valid_logits[task_mask], 
                                        valid_vals[task_mask],
                                        reduction='sum'
                                    )
                                    prop_losses[task_id] += task_loss
                                    prop_counts[task_id] += task_mask.sum()
                        
                        # Collect predictions for AUC/BAC
                        probs = F.softmax(valid_logits, dim=-1)
                        all_preds.append(probs)
                        all_targets.append(valid_vals)

        # Gather predictions across ranks for metrics
        logging.info("Computing overall metrics...")
        predictions, targets = self._gather_predictions_and_targets(all_preds, all_targets)
        overall_metrics = self._compute_metrics(predictions, targets)

        # Reduce across ranks if distributed
        if dist.is_initialized():
            logging.info("Reducing across ranks...")
            total_loss_t = torch.tensor(total_loss, device=self.rank)
            total_samples_t = torch.tensor(total_samples, device=self.rank)
            
            dist.all_reduce(total_loss_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_samples_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(prop_losses, op=dist.ReduceOp.SUM)
            dist.all_reduce(prop_counts, op=dist.ReduceOp.SUM)
            
            total_loss = total_loss_t.item()
            total_samples = total_samples_t.item()

        # Compute final metrics
        overall_loss = total_loss / max(1, total_samples)
        
        # Stratified loss: mean of per-property average losses
        valid_props = (prop_counts > 0)
        if valid_props.any():
            per_prop_avg_losses = prop_losses[valid_props] / prop_counts[valid_props]
            stratified_loss = per_prop_avg_losses.mean().item()
            num_props_seen = valid_props.sum().item()
        else:
            stratified_loss = overall_loss
            num_props_seen = 0

        # Build result dictionary
        result = {
            "loss": stratified_loss,  # Primary metric: stratified loss
            "loss_overall": overall_loss,  # Sample-weighted overall loss
            "auc": overall_metrics["auc"],
            "bac": overall_metrics["bac"],
            "num_properties_evaluated": num_props_seen,
        }
        
        # Add per-property metrics (optional, for monitoring)
        for task_id in range(self.num_tasks):
            if prop_counts[task_id] > 0:
                avg_loss = (prop_losses[task_id] / prop_counts[task_id]).item()
                count = prop_counts[task_id].item()
                result[f"loss_property_{task_id}"] = avg_loss
                result[f"count_property_{task_id}"] = int(count)

        # Logging
        if self.rank == 0:
            logging.info(f"📊 EVALUATION COMPLETE")
            logging.info(f"📊 Stratified Loss: {stratified_loss:.4f} (across {num_props_seen} properties)")
            logging.info(f"📊 Overall Loss: {overall_loss:.4f}")
            logging.info(f"📊 AUC: {overall_metrics['auc']:.4f}, BAC: {overall_metrics['bac']:.4f}")
            
            if num_props_seen > 0:
                # Show properties with highest losses
                prop_losses_cpu = prop_losses.cpu().numpy()
                prop_counts_cpu = prop_counts.cpu().numpy()
                
                valid_indices = np.where(prop_counts_cpu > 0)[0]
                if len(valid_indices) > 0:
                    avg_losses = prop_losses_cpu[valid_indices] / prop_counts_cpu[valid_indices]
                    sorted_indices = valid_indices[np.argsort(avg_losses)[::-1]]
                    
                    logging.info(f"📊 Top 5 most challenging properties:")
                    for i, idx in enumerate(sorted_indices[:5]):
                        avg_loss = prop_losses_cpu[idx] / prop_counts_cpu[idx]
                        count = int(prop_counts_cpu[idx])
                        logging.info(f"   Property {idx}: Loss={avg_loss:.4f} (n={count})")

        return result