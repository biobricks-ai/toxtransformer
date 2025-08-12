import logging
import pathlib
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from sklearn.metrics import roc_auc_score, balanced_accuracy_score

# ---- GradNorm ----
from gradnorm_pytorch import GradNormLossWeighter
from helper.trainer.trainer_core import TrainerCore

class GradNormManager:
    """
    GradNorm over a fixed set of tasks (size = num_total_tasks).
    - Keeps a stable slot for every property via prop_to_index.
    - Absent tasks receive a graph-connected zero loss so gradients are zero,
      preserving weight alignment and avoiding autograd issues.
    - Recommended for sparse batches with many tasks overall.
    """

    def __init__(
        self,
        ddp_model: torch.nn.Module,
        num_total_tasks: int,
        prop_to_index: dict,                 # property_id -> stable [0..num_total_tasks-1]
        lossfn: nn.Module | None = None,
        learning_rate: float = 1e-4,
        restoring_force_alpha: float = 0.0,  # keep 0.0 for sparse batches to reduce drift on absent tasks
        accelerator=None,
    ):
        self.ddp_model = ddp_model
        self.num_total_tasks = int(num_total_tasks)
        self.prop_to_index = prop_to_index
        self.lossfn = lossfn or nn.CrossEntropyLoss()
        self.learning_rate = learning_rate
        self.restoring_force_alpha = restoring_force_alpha
        self.accelerator = accelerator

        m = ddp_model.module if isinstance(ddp_model, DDP) else ddp_model
        if not hasattr(m, "shared_base"):
            raise RuntimeError("Model must expose `.shared_base` for GradNorm (shared parameter).")
        self.shared_param = m.shared_base

        # initialize once with the full number of tasks
        self._weighter = GradNormLossWeighter(
            num_losses=self.num_total_tasks,
            learning_rate=self.learning_rate,
            restoring_force_alpha=self.restoring_force_alpha,
            grad_norm_parameters=self.shared_param,
            accelerator=self.accelerator
        )

    def backward_from_logits(
        self,
        properties: torch.Tensor,  # [B, T]
        values: torch.Tensor,      # [B, T]
        logits: torch.Tensor,      # [B, T, C]
        mask: torch.Tensor,        # [B, T] bool
        grad_accum_scale: int = 1
    ) -> torch.Tensor:
        B, T, C = logits.shape
        logits_f = logits.view(-1, C)
        props_f  = properties.view(-1)
        vals_f   = values.view(-1)
        mask_f   = mask.view(-1)

        # graph-connected zero (same device/dtype, requires grad path)
        zero_proxy = logits_f.sum() * 0.0

        if not mask_f.any():
            (zero_proxy).backward()  # no-op but keeps graph consistent for accum
            return zero_proxy.detach()

        # collect indices per task present this step
        mask_positions = torch.nonzero(mask_f, as_tuple=False).squeeze(1)
        props_present  = props_f[mask_f]

        # bucket valid positions by stable task slot
        # (only iterate over present ones; we’ll fill a dense list later)
        per_task_positions = {}  # slot_idx -> tensor of flat positions
        for local_i, p in enumerate(props_present.tolist()):
            slot = self.prop_to_index.get(p, None)
            if slot is None:
                # optionally: skip unknown tasks or map them to a catch-all slot
                continue
            pos = mask_positions[local_i]
            if slot in per_task_positions:
                per_task_positions[slot].append(pos)
            else:
                per_task_positions[slot] = [pos]

        # build dense per-task loss list of fixed length
        losses = [zero_proxy for _ in range(self.num_total_tasks)]
        for slot, pos_list in per_task_positions.items():
            idx = torch.stack(pos_list, dim=0)
            losses[slot] = self.lossfn(logits_f[idx], vals_f[idx])

        # scale for grad accumulation
        losses = [l * (1.0 / grad_accum_scale) for l in losses]

        # weighted sum + backward (library handles this)
        self._weighter.backward(losses)

        # return mean over *present* task losses (unweighted) for logging
        if per_task_positions:
            mean_loss = torch.stack([losses[s] for s in per_task_positions.keys()]).mean().detach()
        else:
            mean_loss = zero_proxy.detach()
        return mean_loss

    def current_weights(self):
        try:
            return self._weighter.loss_weights.detach().float().cpu().tolist()
        except Exception:
            return None

# ----------------------------
# GradNorm trainer
# ----------------------------
class GradNormTrainer(TrainerCore):
    def __init__(self, *args, gradnorm_cfg: Optional[dict] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.lossfn = nn.CrossEntropyLoss()

        # stable task universe (example)
        # assumes tokenizer has contiguous assay ids 0..N-1
        num_tasks = getattr(self.tokenizer, "num_assays", None)
        assert num_tasks is not None, "tokenizer.num_assays is required"
        prop_to_index = {i: i for i in range(num_tasks)}  # or your custom mapping

        self.gradnorm = GradNormManager(
            ddp_model=self.model,
            num_total_tasks=num_tasks,
            prop_to_index=prop_to_index,
            lossfn=self.lossfn,
            **(gradnorm_cfg or dict(learning_rate=1e-4, restoring_force_alpha=0.0)),
        )

    def _train_batch(self, batch) -> float:
        selfies, properties, values, mask = batch
        if self.global_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()

        selfies = selfies.to(self.rank, non_blocking=True)
        properties = properties.to(self.rank, non_blocking=True)
        values = values.to(self.rank, non_blocking=True)
        mask = mask.to(self.rank, non_blocking=True)

        should_sync = (self.global_step + 1) % self.gradient_accumulation_steps == 0
        context = torch.enable_grad() if should_sync else self.model.no_sync()

        with context:
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = self.model(selfies, properties, values, mask)  # [B,T,C]
                batch_loss = self.gradnorm.backward_from_logits(
                    properties=properties,
                    values=values,
                    logits=logits,
                    mask=mask,
                    grad_accum_scale=self.gradient_accumulation_steps
                )

        if should_sync:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

        # return de-accumulated loss for logging
        return float(batch_loss.detach()) * self.gradient_accumulation_steps

    def gradnorm_weights(self):
        return self.gradnorm.current_weights()
