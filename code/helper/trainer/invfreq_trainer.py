# helper/trainer/invfreq_trainer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging

from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from helper.trainer.trainer_core import TrainerCore


class EMAFreqs:
    """
    Tracks an EMA of per-task prevalence (probability a masked position belongs to task t).
    freqs[t] ~ P(task=t | masked position), smoothed with decay.
    """
    def __init__(self, num_tasks: int, decay: float = 0.99, eps: float = 1e-6, device: torch.device | int = "cpu"):
        self.decay = decay
        self.eps = eps
        self.freqs = torch.full((num_tasks,), fill_value=1.0 / max(1, num_tasks), dtype=torch.float32, device=device)

    @torch.no_grad()
    def update_from_counts(self, task_counts: torch.Tensor, total: int):
        """
        task_counts: Long[ num_tasks ] counts in current batch (on same device as freqs)
        total: int, total masked positions in batch
        """
        if total <= 0:
            return
        # batch prevalence vector (sums to 1 over tasks that appeared)
        p = task_counts.to(self.freqs.dtype) / float(total)
        # EMA update
        self.freqs.mul_(self.decay).add_(p * (1.0 - self.decay))

    @torch.no_grad()
    def inverse_weights(self, normalize: bool = True, clamp_max: float | None = 10.0):
        w = 1.0 / (self.eps + self.freqs)
        if normalize:
            w = w / w.mean().clamp_min(1e-8)  # Normalize first
        if clamp_max is not None:
            w = torch.clamp(w, max=clamp_max)  # Then clamp
        return w


class InverseFrequencyWeightedTrainer(TrainerCore):
    """
    Weights per-task losses by inverse EMA prevalence.
    - No external deps.
    - Stable task indexing via tokenizer.assay_indexes().
    """

    def __init__(self,
                 *args,
                 ema_decay: float = 0.99,
                 ema_eps: float = 1e-6,
                 max_weight: float | None = 10.0,
                 log_weights_every: int = 100,
                 update_weights_during_training: bool = False,
                 batch_level_weighting: bool = False,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.lossfn = nn.CrossEntropyLoss()
        self.max_weight = max_weight
        self.log_weights_every = log_weights_every
        self.update_weights_during_training = update_weights_during_training
        self.batch_level_weighting = batch_level_weighting

        # Build stable 0..N-1 mapping from raw property tokens
        assay_tokens = sorted(self.tokenizer.assay_indexes().values())
        self._minprop = assay_tokens[0]
        self._num_tasks = len(assay_tokens)
        # For speed we'll derive slot with (prop - minprop). If your ids aren't truly contiguous,
        # replace with an explicit dict: {raw_id: slot}
        self._contiguous = (assay_tokens == list(range(self._minprop, self._minprop + self._num_tasks)))
        
        if not self._contiguous:
            # fallback to explicit map
            self._prop_to_index = {p: i for i, p in enumerate(assay_tokens)}
        else:
            self._prop_to_index = None  # use offset math

        device = self.rank
        self.ema = EMAFreqs(self._num_tasks, decay=ema_decay, eps=ema_eps, device=device)

        logging.info(f"""Initialized InverseFrequencyWeightedTrainer with:
        - {self._num_tasks} tasks
        - {self.max_weight}""")

    def _prop_to_slot(self, props_flat_valid: torch.Tensor) -> torch.Tensor:
        """
        Map raw property tokens -> stable slot [0..num_tasks-1]
        props_flat_valid: 1D tensor of raw ids for masked positions
        """
        if self._contiguous:
            return (props_flat_valid - self._minprop).clamp_min(0)
        # explicit map
        # Vectorized map via CPU list + tensor rewrap (frequency small per step)
        mapped = [self._prop_to_index.get(int(p), -1) for p in props_flat_valid.tolist()]
        out = torch.tensor(mapped, device=props_flat_valid.device, dtype=torch.long)
        return out

    def _log_weight_info(self, counts: torch.Tensor, weights: torch.Tensor):
        """Enhanced logging for debugging"""
        if self.rank == 0 and self.global_step % self.log_weights_every == 0:
            active_mask = counts > 0
            active_count = active_mask.sum().item()
            
            if active_count > 0:
                active_weights = weights[active_mask]
                active_freqs = self.ema.freqs[active_mask]
                
                # Add this debug info:
                raw_inverse = 1.0 / (self.ema.eps + active_freqs)
                normalized = raw_inverse / raw_inverse.mean()
                
                logging.info(f"Step {self.global_step}: {active_count}/{self._num_tasks} tasks active")
                logging.info(f"  Raw inverse: min={raw_inverse.min():.1f}, max={raw_inverse.max():.1f}")
                logging.info(f"  After norm: min={normalized.min():.3f}, max={normalized.max():.3f}")
                logging.info(f"  Final weights: min={active_weights.min():.3f}, max={active_weights.max():.3f}")
                logging.info(f"  Freqs: min={active_freqs.min():.6f}, max={active_freqs.max():.6f}")

    @torch.no_grad()
    def warmup_frequencies(self, warmup_steps: int = 100, dataloader=None, cache_path: str = None):
        """
        Run warmup to estimate initial property frequencies before training starts.
        
        Args:
            warmup_steps: Number of batches to sample
            dataloader: Optional dataloader to use. If None, uses self.trn_iterator
            cache_path: Optional path to save/load frequency cache
        """
        import pickle
        import hashlib
        
        # Try to load from cache first
        if cache_path:
            cache_key = f"{self._num_tasks}_{warmup_steps}_{self.max_weight}"
            cache_file = f"{cache_path}/freq_cache_{cache_key}.pkl"
            
            try:
                with open(cache_file, 'rb') as f:
                    cached_freqs = pickle.load(f)
                    self.ema.freqs = torch.tensor(cached_freqs, device=self.rank, dtype=torch.float32)
                    
                if self.rank == 0:
                    freqs = self.ema.freqs.cpu().numpy()
                    weights = self.ema.inverse_weights().cpu().numpy()
                    logging.info(f"📂 Loaded cached frequencies from {cache_file}")
                    logging.info(f"📊 Task frequencies: min={freqs.min():.4f}, max={freqs.max():.4f}, mean={freqs.mean():.4f}")
                    logging.info(f"⚖️  Weights: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
                return
            except (FileNotFoundError, Exception) as e:
                if self.rank == 0:
                    logging.info(f"Cache miss or error ({e}), running warmup...")
        
        # Run warmup - ultra memory efficient
        dl = dataloader if dataloader is not None else self.trn_iterator
        
        if self.rank == 0:
            source = "custom dataloader" if dataloader is not None else "training iterator"
            logging.info(f"🔥 Starting frequency warmup for {warmup_steps} steps using {source}...")
        
        total_counts = torch.zeros(self._num_tasks, device='cpu', dtype=torch.long)  # Keep on CPU
        total_positions = 0
        
        for step, batch in enumerate(dl):
            if step >= warmup_steps:
                break
                
            # Process immediately and release
            _, properties, _, mask = batch
            
            # Work on CPU to save GPU memory
            mask_f = mask.view(-1).cpu()
            if mask_f.any():
                properties_f = properties.view(-1).cpu()
                pos = torch.nonzero(mask_f, as_tuple=False).squeeze(1)
                props_valid = properties_f[pos]
                
                # Convert to slots on CPU
                if self._contiguous:
                    slots = (props_valid - self._minprop).clamp_min(0)
                else:
                    mapped = [self._prop_to_index.get(int(p), -1) for p in props_valid.tolist()]
                    slots = torch.tensor(mapped, dtype=torch.long)
                
                valid_mask = (slots >= 0) & (slots < self._num_tasks)
                slots = slots[valid_mask]
                
                counts = torch.bincount(slots, minlength=self._num_tasks)
                total_counts += counts
                total_positions += pos.numel()
            
            # Explicitly delete batch data
            del batch, properties, mask, mask_f
            if 'properties_f' in locals():
                del properties_f, pos, props_valid, slots, counts
            
            # Aggressive cleanup
            if step % 100 == 0:
                torch.cuda.empty_cache()
        
        # Update EMA with accumulated statistics
        if total_positions > 0:
            self.ema.update_from_counts(total_counts.to(self.rank), total_positions)
            
            # Save to cache
            if cache_path and self.rank == 0:
                import os
                os.makedirs(cache_path, exist_ok=True)
                with open(cache_file, 'wb') as f:
                    pickle.dump(self.ema.freqs.cpu().numpy(), f)
                logging.info(f"💾 Saved frequencies to {cache_file}")
            
            freqs = self.ema.freqs.cpu().numpy()
            weights = self.ema.inverse_weights().cpu().numpy()
            logging.info(f"✅ Warmup complete: {total_positions} positions across {warmup_steps} steps")
            logging.info(f"📊 Task frequencies: min={freqs.min():.4f}, max={freqs.max():.4f}, mean={freqs.mean():.4f}")
            logging.info(f"⚖️  Initial weights: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
        
        # Final cleanup
        del total_counts
        torch.cuda.empty_cache()

    def _compute_batch_loss(self, logits, values, mask, properties):
        """Compute weighted loss for a batch, return (loss_tensor, batch_weight, raw_loss)"""
        B, T, C = logits.shape
        logits_f = logits.view(-1, C)
        vals_f = values.view(-1)
        mask_f = mask.view(-1)

        if not mask_f.any():
            zero = logits_f.sum() * 0.0
            return zero, 0.0, 0.0

        # Get valid positions and properties
        pos = torch.nonzero(mask_f, as_tuple=False).squeeze(1)
        props_valid = properties.view(-1)[pos]
        slots = self._prop_to_slot(props_valid)

        valid_mask = (slots >= 0) & (slots < self._num_tasks)
        slots = slots[valid_mask]
        pos = pos[valid_mask]

        counts = torch.bincount(slots, minlength=self._num_tasks)
        total = int(pos.numel())

        # Update EMA if enabled
        if self.update_weights_during_training:
            self.ema.update_from_counts(counts.to(self.rank), total)

        # Get inverse-frequency weights
        weights = self.ema.inverse_weights(normalize=True, clamp_max=self.max_weight)
        self._log_weight_info(counts, weights)

        # Compute per-task losses
        unique_slots = torch.nonzero(counts, as_tuple=False).squeeze(1)
        if len(unique_slots) == 0:
            zero = logits_f.sum() * 0.0
            return zero, 0.0, 0.0

        per_slot_losses = []
        per_slot_weights = []
        for s in unique_slots.tolist():
            idx = slots == s
            idx_pos = pos[idx]
            loss_s = self.lossfn(logits_f[idx_pos], vals_f[idx_pos])
            per_slot_losses.append(loss_s)
            per_slot_weights.append(weights[s])

        ls = torch.stack(per_slot_losses)
        ws = torch.stack(per_slot_weights).to(ls.dtype)
        
        # Batch weight is mean of task weights in this batch
        batch_weight = ws.mean().item()
        # For stronger batch weighting, could use: batch_weight = ws.sum().item()
        weighted_loss = (ls * ws).sum() / ws.sum().clamp_min(1e-8)
        raw_loss = float(ls.mean().detach())

        return weighted_loss, batch_weight, raw_loss

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
                logits = self.model(selfies, properties, values, mask)
                weighted_loss, batch_weight, raw_loss = self._compute_batch_loss(logits, values, mask, properties)

                if self.batch_level_weighting:
                    # Apply batch weight as a simple scalar multiplier
                    batch_scalar = batch_weight  # Mean of task weights in this batch
                    final_loss = weighted_loss * batch_scalar / self.gradient_accumulation_steps
                    final_loss.backward()
                    batch_loss_scalar = raw_loss
                else:
                    # Standard approach: backward immediately
                    (weighted_loss / self.gradient_accumulation_steps).backward()
                    batch_loss_scalar = raw_loss

        if should_sync:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            self.scheduler.step()

        return batch_loss_scalar