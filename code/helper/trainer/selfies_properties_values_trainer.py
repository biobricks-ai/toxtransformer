from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import torch
import torch.distributed as dist
import torch.nn as nn

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging
import pathlib
import numpy as np
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import traceback
import psutil
import datetime
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from typing import Optional, Tuple, Dict
from helper.trainer.trainer_core import TrainerCore

class SelfiesPropertiesValuesTrainer(TrainerCore):
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.lossfn = nn.CrossEntropyLoss()

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
                B, T, C = logits.shape
                logits_f = logits.view(-1, C)
                vals_f   = values.view(-1)
                mask_f   = mask.view(-1)
                if mask_f.any():
                    loss = self.lossfn(logits_f[mask_f], vals_f[mask_f]) / self.gradient_accumulation_steps
                else:
                    loss = logits_f.new_zeros(())

            loss.backward()

        if should_sync:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

        return float(loss.detach()) * self.gradient_accumulation_steps
