"""
RestrictedTrainer - A specialized trainer for evaluating specific property tokens.

This module extends the base Trainer class with property-specific evaluation capabilities,
focusing loss calculation and metrics only on the specified property tokens.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from helper.trainer.trainer import Trainer
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)

class RestrictedEvalMixin:
    """
    A mixin class that restricts evaluation metrics to specific property tokens.
    Assumes property-value pairs in sequential positions (<p1> <v1> <p2> <v2> etc).
    """

    def set_target_properties(self, target_properties=None):
        self.target_properties = target_properties
        self.log(f"Rank {self.rank}: Target properties set to {self.target_properties}")
        return self
    
    def set_evaluation_positions(self, eval_positions=None):
        self.eval_positions = eval_positions if eval_positions else list(range(self.nprops))
        self.eval_positions = [p * 2 + 2 for p in self.eval_positions]
        self.log(f"Rank {self.rank}: Restricted evaluation set to properties {self.eval_positions}")
        return self
    
    def filter_values(self, tensor, positions):
        return tensor[:, positions]
    
    def filter_properties(self, properties, sequence_tensor, prediction_tensor):
        key_mask = torch.isin(sequence_tensor, torch.tensor(properties, device=sequence_tensor.device))
        return prediction_tensor[key_mask]

    def _eval_generator(self, value_tokens_to_binary, value_indexes):

        for i, (inp, teach, out) in enumerate(self.valdl):
            if self.eval_samples is not None and i >= self.eval_samples:
                break

            inp, teach, raw_out = inp.to(self.rank), teach.to(self.rank), out.to(self.rank)
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    raw_pred = self.model(inp, teach)
                    # loss = self.lossfn(self.model.module.parameters(), raw_pred, raw_out)
                    # TODO get loss working
                    loss = 0.
                    
                    out = raw_out[:, self.eval_positions]
                    prd = raw_pred[:, self.eval_positions, :][:, :, value_indexes]

                    # Map token IDs in out to 0/1 using the lookup
                    out_mapped = out.clone()
                    for token, label in value_tokens_to_binary.items():
                        out_mapped[out == token] = label

                    yield out_mapped, prd, loss

    def _gather_metric(self, metric):
        device = torch.device(f"cuda:{self.rank}")
        metric_tensor = torch.tensor([float(metric)], device=device)
        dist.all_reduce(metric_tensor, op=dist.ReduceOp.SUM)
        return metric_tensor.item() / dist.get_world_size()

    def _eval_all(self, max_eval_batches: Optional[int] = None) -> Tuple[float, float, float]:
        self.model.eval()
        
        value_tokens_to_binary = {v: i for i, v in enumerate(self.tokenizer.value_indexes().values())}
        value_indexes = list(value_tokens_to_binary.keys())
        outpreds = [(out, pred, loss) for out, pred, loss in self._eval_generator(value_tokens_to_binary, value_indexes)]
        dist.barrier()

        if not outpreds:
            return 0.0, 0.0, float("inf")

        outs = [out for out, _, _ in outpreds]
        preds = [pred for _, pred, _ in outpreds]
        losses = [loss for _, _, loss in outpreds]

        # Concatenate tensors
        out = torch.cat(outs, dim=0)        # shape: [N, T]
        pred = torch.cat(preds, dim=0)      # shape: [N, T, V_restricted]
        probs = F.softmax(pred, dim=-1)     # shape: [N, T, V_restricted]
        pred_labels = torch.argmax(probs, dim=-1)  # shape: [N, T]

        # Defensive shape check
        if out.shape != pred_labels.shape:
            raise ValueError(f"Inconsistent shapes: out={out.shape}, pred_labels={pred_labels.shape}")

        # Move to CPU
        y_true = out.flatten().cpu().numpy()
        y_pred = pred_labels.flatten().cpu().numpy()
        probs_1 = probs[:, :, 1].flatten().cpu().numpy()

        logger.info(f"y_true: {y_true[:10]}")
        logger.info(f"y_pred: {y_pred[:10]}")
        logger.info(f"probs_1: {probs_1[:10]}")

        # Metrics
        rank_bac = balanced_accuracy_score(y_true, y_pred)
        rank_auc = roc_auc_score(y_true, probs_1)
        rank_loss = np.mean(losses)

        # All-reduce
        bac = self._gather_metric(rank_bac)
        auc = self._gather_metric(rank_auc)
        loss = self._gather_metric(rank_loss)

        return {'loss': loss, 'auc': auc, 'bac': bac}


def create_restricted_trainer(
    model, rank, tokenizer, train_iterator, batch_size, 
    target_properties, target_positions, nprops,
    scheduler_warmup_steps=10000, scheduler_max_steps=100000, max_steps=100000,
    savepath=None, metrics_path=None, validation_dataloader=None, first_eval=10, eval_every=10000, eval_samples=400,
    scheduler_min_lr=1e-6, scheduler_max_lr=3e-4
):
    """Factory function to create and configure a RestrictedTrainer instance."""
    trainer = RestrictedTrainer(
        model=model,
        rank=rank,
        tokenizer=tokenizer,
        trn_iterator=train_iterator,
        batch_size=batch_size,
        target_properties=target_properties,
        target_positions=target_positions,
        nprops=nprops,
        scheduler_warmup_steps=scheduler_warmup_steps,
        scheduler_max_steps=scheduler_max_steps,
        max_steps=max_steps,
        first_eval=first_eval,
        eval_every=eval_every,
        eval_samples=eval_samples,
        scheduler_min_lr=scheduler_min_lr,
        scheduler_max_lr=scheduler_max_lr
    )

    if savepath:
        trainer.set_model_savepath(savepath)

    if metrics_path:
        trainer.set_metrics_file(metrics_path)

    if validation_dataloader:
        trainer.set_validation_dataloader(validation_dataloader)

    return trainer

class RestrictedTrainer(RestrictedEvalMixin, Trainer):
    """
    A specialized trainer that extends the base Trainer class with restricted evaluation capabilities.
    """
    def __init__(self, *args, **kwargs):
        self.target_properties = kwargs.pop('target_properties', None)
        self.target_positions = kwargs.pop('target_positions', None)
        self.nprops = kwargs.pop('nprops', None)

        super().__init__(*args, **kwargs)

        # Optional: set evaluation positions immediately if available
        if self.target_positions is not None:
            self.set_evaluation_positions(self.target_positions)

