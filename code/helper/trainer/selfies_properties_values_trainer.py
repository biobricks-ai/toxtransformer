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


class WarmupCosineScheduler:
    """
    Creates a learning rate scheduler with linear warmup followed by cosine annealing with restarts.
    """
    
    def __init__(self, optimizer, warmup_steps, cosine_cycle_length, min_lr, max_lr, cosine_t_mult=2):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of steps for linear warmup
            cosine_cycle_length: Length of first cosine cycle (T_0)
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate (optimizer should be initialized with this)
            cosine_t_mult: Multiplier for subsequent cycle lengths
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.cosine_cycle_length = cosine_cycle_length
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cosine_t_mult = cosine_t_mult
        
        # Create the scheduler
        self._build_scheduler()
    
    def _build_scheduler(self):
        """Build the composite scheduler."""
        # 1. Linear warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=self.min_lr / self.max_lr,  # Start at min_lr
            end_factor=1.0,                          # End at max_lr  
            total_iters=self.warmup_steps
        )
        
        # 2. Cosine annealing with warm restarts
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.cosine_cycle_length,
            T_mult=self.cosine_t_mult,
            eta_min=self.min_lr
        )
        
        # 3. Combine them
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )
    
    def step(self):
        """Step the scheduler."""
        self.scheduler.step()
    
    def get_last_lr(self):
        """Get the last learning rate."""
        return self.scheduler.get_last_lr()
    
    def state_dict(self):
        """Get scheduler state dict."""
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load scheduler state dict."""
        self.scheduler.load_state_dict(state_dict)


class SelfiesPropertiesValuesTrainer:
    
    def __init__(self, model, rank, tokenizer, trn_iterator, batch_size, 
                 scheduler, max_steps=100000, first_eval=100, eval_every=10000, 
                 eval_samples=400, effective_accum_batch_size=1024, find_unused_parameters=False):

        self.rank = rank
        self.global_step = 0
        self.trn_iterator = trn_iterator
        self.tokenizer = tokenizer

        torch.cuda.set_device(rank)
        logging.info(f"Rank {self.rank}: Setting CUDA device to {rank}")
        self.model = model.to(rank)
        self.lossfn = nn.CrossEntropyLoss()
        
        dist.barrier(device_ids=[self.rank])
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=find_unused_parameters)
        self.max_steps = max_steps

        # Gradient accumulation setup
        self.gradient_accumulation_steps = max(1, effective_accum_batch_size // (batch_size * dist.get_world_size()))
        
        # Calculate effective batch sizes
        effective_batch_size_per_gpu = batch_size * self.gradient_accumulation_steps
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        total_effective_batch_size = effective_batch_size_per_gpu * world_size

        # Use the provided scheduler
        self.scheduler = scheduler
        self.optimizer = scheduler.optimizer  # Get optimizer from scheduler
        
        # Training setup
        self.metrics_path = None
        self.best_loss = np.inf
        self.scaler = GradScaler()
        self.first_eval = first_eval
        self.eval_every = eval_every
        self.eval_samples = eval_samples

        logging.info(f"""Trainer initialized with:
        - Model: {self.model.__class__.__name__}
        - Rank: {self.rank}
        - Training samples: {len(self.trn_iterator.dataset)}
        - Evaluation samples: {self.eval_samples}
        - Effective batch size per GPU: {effective_batch_size_per_gpu}
        - Total effective batch size: {total_effective_batch_size}
        - Gradient accumulation steps: {self.gradient_accumulation_steps}
        """)

    def set_model_savepath(self, savepath):
        self.savepath = pathlib.Path(savepath)
        self.savepath.mkdir(exist_ok=True, parents=True)
        logging.info(f"Rank {self.rank}: Model save path set to {self.savepath}")
        return self

    def set_validation_dataloader(self, valdl):
        self.valdl = valdl
        logging.info(f"Rank {self.rank}: Validation dataloader set.")
        return self

    def set_metrics_file(self, metrics_path, overwrite=False):
        self.metrics_path = pathlib.Path(metrics_path)
        if self.rank == 0:
            self.metrics_path.parent.mkdir(exist_ok=True, parents=True)
            if overwrite:
                with open(self.metrics_path, 'w') as f:
                    f.write("type\tbatch\tloss\tlr\tauc\tbac\n")
        logging.info(f"Rank {self.rank}: Metrics file path set to {self.metrics_path}")
        return self

    def _train_batch(self, selfies, properties, values, mask):
        """Train on a single batch."""
        # Only zero gradients at the beginning of accumulation cycle
        if self.global_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()

        # Move data to device
        selfies = selfies.to(self.rank, non_blocking=True)
        properties = properties.to(self.rank, non_blocking=True)
        values = values.to(self.rank, non_blocking=True)
        mask = mask.to(self.rank, non_blocking=True)

        should_sync = (self.global_step + 1) % self.gradient_accumulation_steps == 0
        context = torch.enable_grad() if should_sync else self.model.no_sync()
        
        # Forward pass and loss calculation
        with context:
            with autocast(device_type='cuda', dtype=torch.float16):
                pred = self.model(selfies, properties, values, mask)  # [batch_size, num_props, num_value_classes]
                
                # Reshape for loss calculation
                pred_flat = pred.view(-1, pred.size(-1))
                values_flat = values.view(-1)
                mask_flat = mask.view(-1)

                # Calculate loss only on valid positions
                if mask_flat.any():
                    loss = self.lossfn(pred_flat[mask_flat], values_flat[mask_flat])
                else:
                    loss = torch.tensor(0.0, device=pred.device, requires_grad=True)
                
                loss = loss / self.gradient_accumulation_steps

            # Scale gradients and accumulate
            self.scaler.scale(loss).backward()

        # Only update weights at the end of accumulation cycle
        if should_sync:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

        self.global_step += 1
        return loss.detach().item() * self.gradient_accumulation_steps

    def _eval_all(self, max_eval_batches: Optional[int] = None) -> Dict[str, float]:
        """Evaluate model."""
        torch.cuda.empty_cache()
        max_eval_batches = self.eval_samples if max_eval_batches is None else max_eval_batches
        max_eval_batches = min(max_eval_batches, len(self.valdl))

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds = []
        all_targets = []

        dist.barrier(device_ids=[self.rank])
        for i, (selfies, properties, values, mask) in enumerate(self.valdl):
            if max_eval_batches is not None and i >= max_eval_batches:
                break

            selfies = selfies.to(self.rank, non_blocking=True)
            properties = properties.to(self.rank, non_blocking=True) 
            values = values.to(self.rank, non_blocking=True)
            mask = mask.to(self.rank, non_blocking=True)
            
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):
                    pred = self.model(selfies, properties, values, mask)
                    
                    # Calculate loss using mask
                    pred_flat = pred.view(-1, pred.size(-1))
                    values_flat = values.view(-1)
                    mask_flat = mask.view(-1)
                    
                    if mask_flat.any():
                        loss = self.lossfn(pred_flat[mask_flat], values_flat[mask_flat])
                    else:
                        loss = torch.tensor(0.0, device=pred.device)

                # Collect predictions for metrics
                pred_probs = F.softmax(pred, dim=-1)
                pred_probs_flat = pred_probs.view(-1, pred_probs.size(-1))
                
                if mask_flat.any():
                    all_preds.append(pred_probs_flat[mask_flat])
                    all_targets.append(values_flat[mask_flat])

                total_loss += loss.item()
                num_batches += 1

        # Aggregate loss across ranks
        total_loss_tensor = torch.tensor(total_loss, device=self.rank)
        num_batches_tensor = torch.tensor(num_batches, device=self.rank)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_batches_tensor, op=dist.ReduceOp.SUM)

        mean_loss = total_loss_tensor.item() / num_batches_tensor.item() if num_batches_tensor.item() > 0 else 0.0

        # Calculate metrics
        if all_preds:
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Gather across ranks
            local_len = torch.tensor([all_preds.size(0)], device=self.rank)
            lengths = [torch.zeros_like(local_len) for _ in range(dist.get_world_size())]
            dist.all_gather(lengths, local_len)
            max_len = max(l.item() for l in lengths)

            # Pad and gather
            pad = max_len - local_len.item()
            if pad > 0:
                all_preds = F.pad(all_preds, (0, 0, 0, pad))
                all_targets = F.pad(all_targets, (0, pad), value=-1)

            gathered_preds = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
            gathered_targets = [torch.zeros_like(all_targets) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_preds, all_preds)
            dist.all_gather(gathered_targets, all_targets)

            # Convert to numpy and calculate metrics
            true_total = sum(l.item() for l in lengths)
            all_preds = torch.cat(gathered_preds, dim=0)[:true_total].cpu().numpy()
            all_targets = torch.cat(gathered_targets, dim=0)[:true_total].cpu().numpy()

            # Filter out any padding values (-1) that might have slipped through
            valid_mask = all_targets >= 0
            all_preds = all_preds[valid_mask]
            all_targets = all_targets[valid_mask]

            # Check if we have valid data and exactly 2 classes for binary classification
            unique_targets = np.unique(all_targets)
            if len(all_targets) > 0 and len(unique_targets) == 2 and all_preds.shape[1] >= 2:
                try:
                    auc = roc_auc_score(all_targets, all_preds[:, 1])
                    bac = balanced_accuracy_score(all_targets, all_preds.argmax(axis=1))
                except Exception as e:
                    logging.warning(f"Rank {self.rank}: Error calculating metrics: {e}")
                    auc = 0.0
                    bac = 0.0
            else:
                logging.info(f"Rank {self.rank}: Skipping metrics - targets: {len(all_targets)}, unique: {len(unique_targets) if len(all_targets) > 0 else 0}, pred_shape: {all_preds.shape}")
                auc = 0.0
                bac = 0.0
        else:
            auc = 0.0
            bac = 0.0
        
        return {'loss': mean_loss, 'auc': auc, 'bac': bac}

    def start(self):
        """Start training loop."""
        logging.info(f"Rank {self.rank}: Starting training loop.")
        epoch = 0

        while self.global_step < self.max_steps:
            self.trn_iterator.sampler.set_epoch(epoch)
            logging.info(f"Rank {self.rank}: Starting epoch {epoch}. step {self.global_step}.")
            dist.barrier(device_ids=[self.rank])

            self.model.train()

            for i, (selfies, properties, values, mask) in enumerate(self.trn_iterator):
                loss = self._train_batch(selfies, properties, values, mask)

                # Log training metrics
                if self.global_step % self.gradient_accumulation_steps == 0 and self.rank == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    with open(self.metrics_path, 'a') as f:
                        f.write(f"train\t{self.global_step}\t{loss:.4f}\t{current_lr:.6f}\t0.0\t0.0\n")
                    logging.info(f"Epoch: {epoch}, Step: {self.global_step}, Train Loss: {loss:.4f}, LR: {current_lr:.6f}")

                # Evaluation
                if self.global_step == self.first_eval or (self.global_step + self.first_eval) % self.eval_every == 0:
                    logging.info(f"Rank {self.rank}: Starting evaluation at step {self.global_step}")
                    torch.cuda.synchronize()
                    dist.barrier(device_ids=[self.rank])

                    self.model.eval()
                    with torch.no_grad():
                        evals = self._eval_all(max_eval_batches=self.eval_samples)
                        eval_loss, auc, bac = evals['loss'], evals['auc'], evals['bac']
                        logging.info(f"Rank {self.rank}: Evaluation complete. Loss: {eval_loss:.4f}, AUC: {auc:.4f}, BAC: {bac:.4f}")

                    if self.rank == 0 and eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.model.module.save(self.savepath)
                        logging.info(f"Rank {self.rank}: New best eval loss ({self.best_loss:.4f}), saving best model")

                    # Periodic save
                    if self.rank == 0:
                        periodic_save_path = self.savepath.parent / f"step_{self.global_step}"
                        periodic_save_path.mkdir(exist_ok=True, parents=True)
                        self.model.module.save(periodic_save_path)

                    # Log evaluation metrics
                    if self.rank == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        with open(self.metrics_path, 'a') as f:
                            f.write(f"eval\t{self.global_step}\t{eval_loss:.4f}\t{current_lr:.6f}\t{auc:.4f}\t{bac:.4f}\n")

                    dist.barrier(device_ids=[self.rank])
                    self.model.train()
            
            epoch += 1

        logging.info(f"Rank {self.rank}: Training loop finished after {self.global_step} steps.")