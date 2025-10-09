import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import random

from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

from helper.trainer.trainer_core import TrainerCore

import numpy as np


class AccumulatedStratifiedPropertyWeightedTrainer(TrainerCore):
    """
    Trainer that computes per-property EMA of training losses and uses
    those EMAs to derive per-property weights for subsequent batches.

    Randomly samples x% of properties and applies calculated dynamic weights
    to them (range: min_score to max_score), with higher sampling probability 
    for poorly performing properties. Non-sampled properties receive a 
    configurable baseline weight (default: 1.0).

    Setting unsampled_property_weight to:
    - 1.0 (default): Importance weighting - all properties train, sampled get emphasis
    - 0.0: Dropout-style - only sampled properties contribute to loss
    - 0.0 < x < 1.0: Reduced contribution from non-sampled properties
    - > 1.0: Enhanced contribution from non-sampled properties (unusual but possible)

    ...
    """

    def __init__(
        self,
        *args,
        ema_alpha: float = 0.2,
        min_score: float = 0.1,
        max_score: float = 10.0,
        skip_initial_batches: int = 10,
        min_properties_for_weighting: Optional[int] = None,
        label_smoothing: float = 0.1,
        log_gpu_mem: bool = False,
        normalize_weights: bool = True,
        acc_coalesce_every: int = 8,
        property_sample_rate: float = 0.3,
        sample_bias_strength: float = 2.0,
        sampling_warmup: int = 50,
        unsampled_property_weight: float = 1.0,  # NEW parameter
        random_seed: Optional[int] = None,
        **kwargs,
    ):
    
        super().__init__(*args, **kwargs)

        # --- Config
        self._ema_alpha = float(ema_alpha)
        self._min_score = float(min_score)
        self._max_score = float(max_score)
        self.unsampled_property_weight = float(unsampled_property_weight)

        self._skip_initial_batches = int(skip_initial_batches)
        self._min_properties_for_weighting = min_properties_for_weighting
        self.log_gpu_mem = bool(log_gpu_mem)
        self.normalize_weights = bool(normalize_weights)
        self._acc_coalesce_every = int(acc_coalesce_every)
        self._acc_windows_seen = 0

        # NEW: Random sampling config
        self.property_sample_rate = float(property_sample_rate)
        self.sample_bias_strength = float(sample_bias_strength)
        self.sampling_warmup = int(sampling_warmup)
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)

        # --- Running stats for normalization (kept as CPU floats)
        self._train_loss_stats_ema = {"mean": 0.0, "std": 1.0}
        self._train_stats_initialized = False

        # --- Caches/State
        self._batches_processed = 0
        self._cached_score_map: Dict[int, float] = {}

        # GPU-side EMA tensors (aligned by _ema_prop_ids)
        self._ema_prop_ids: Optional[torch.Tensor] = None  # [K] long
        self._ema_vals: Optional[torch.Tensor] = None      # [K] float
        self._prop_counts: Optional[torch.Tensor] = None   # [K] float

        # Cached sorted view for fast weight lookups
        self._ema_prop_ids_sorted: Optional[torch.Tensor] = None  # [K] long
        self._ema_scores_sorted: Optional[torch.Tensor] = None    # [K] float

        # Accumulation buffers across grad-accum window (GPU)
        self._acc_prop_ids: Optional[torch.Tensor] = None   # [K*] long
        self._acc_loss_sums: Optional[torch.Tensor] = None  # [K*] float
        self._acc_count_sums: Optional[torch.Tensor] = None # [K*] float
        self._acc_stats_vals: Optional[torch.Tensor] = None # [M*] float

        # Loss (use CE with label smoothing; reduction handled manually)
        self.lossfn = nn.CrossEntropyLoss(label_smoothing=label_smoothing, reduction="none")

        logging.info(
            "Initialized AccumulatedStratifiedPropertyWeightedTrainer "
            f"(ema_alpha={self._ema_alpha}, score_range=[{self._min_score},{self._max_score}], "
            f"skip_initial_batches={self._skip_initial_batches}, "
            f"min_properties_for_weighting={self._min_properties_for_weighting}, "
            f"normalize_weights={self.normalize_weights}, acc_coalesce_every={self._acc_coalesce_every}, "
            f"property_sample_rate={self.property_sample_rate}, sample_bias_strength={self.sample_bias_strength}, "
            f"sampling_warmup={self.sampling_warmup})"
            f"unsampled_property_weight={self.unsampled_property_weight})"
        )

    # --------------------------- Scheduler helpers ---------------------------

    def _is_plateau_phase(self) -> bool:
        phase = getattr(self.scheduler, "_phase", None)
        return phase == "plateau"

    def step_scheduler(self, val_metric: float):
        if self._is_plateau_phase():
            logging.info("Stepping scheduler in plateau phase")
            self.scheduler.step(val_metric)
        else:
            logging.info("Stepping scheduler in non-plateau phase")
            self.scheduler.step()

    # --------------------------- Weighting gate ---------------------------

    def _num_props_with_ema(self) -> int:
        return int(self._ema_prop_ids.numel()) if self._ema_prop_ids is not None else 0

    def _should_skip_batch(self) -> bool:
        if self._batches_processed < self._skip_initial_batches:
            return True
        if self._min_properties_for_weighting is not None:
            if self._num_props_with_ema() < self._min_properties_for_weighting:
                return True
        return False

    def _should_apply_random_sampling(self) -> bool:
        """Check if we should apply random sampling based on warmup period."""
        return self._batches_processed >= self.sampling_warmup

    # --------------------------- Running stats (CPU floats) ---------------------------

    def _update_train_stats(self, current_losses_tensor: torch.Tensor):
        if current_losses_tensor is None or current_losses_tensor.numel() == 0:
            return
        cur_mean = float(current_losses_tensor.mean().item())
        cur_std = float(current_losses_tensor.std(unbiased=False).clamp_min(0.1).item())
        if not self._train_stats_initialized:
            self._train_loss_stats_ema["mean"] = cur_mean
            self._train_loss_stats_ema["std"] = cur_std
            self._train_stats_initialized = True
        else:
            decay = 0.7
            self._train_loss_stats_ema["mean"] = decay * self._train_loss_stats_ema["mean"] + (1 - decay) * cur_mean
            self._train_loss_stats_ema["std"] = decay * self._train_loss_stats_ema["std"] + (1 - decay) * cur_std

    # --------------------------- Random sampling with bias ---------------------------

    @torch.no_grad()
    def _sample_properties_with_bias(self, prop_ids: torch.Tensor, ema_vals: torch.Tensor) -> torch.Tensor:
        """
        Randomly sample properties with bias toward higher losses (worse performance).
        
        Args:
            prop_ids: [K] tensor of property IDs
            ema_vals: [K] tensor of EMA loss values
            
        Returns:
            [K] boolean mask indicating which properties are selected
        """
        K = prop_ids.numel()
        if K == 0:
            return torch.zeros(0, dtype=torch.bool, device=prop_ids.device)
            
        num_to_sample = max(1, int(K * self.property_sample_rate))
        
        # Convert losses to sampling probabilities (higher loss = higher probability)
        # Normalize losses to [0, 1] range, then apply bias
        min_loss = ema_vals.min()
        max_loss = ema_vals.max()
        if max_loss > min_loss:
            normalized_losses = (ema_vals - min_loss) / (max_loss - min_loss)
        else:
            normalized_losses = torch.ones_like(ema_vals)
        
        # Apply bias: higher bias_strength means more focus on worse properties
        sampling_probs = torch.pow(normalized_losses, self.sample_bias_strength)
        sampling_probs = sampling_probs / sampling_probs.sum().clamp_min(1e-8)
        
        # Sample without replacement
        try:
            sampled_indices = torch.multinomial(
                sampling_probs, 
                num_samples=min(num_to_sample, K), 
                replacement=False
            )
            mask = torch.zeros(K, dtype=torch.bool, device=prop_ids.device)
            mask[sampled_indices] = True
        except RuntimeError:
            # Fallback to uniform sampling if multinomial fails
            indices = torch.randperm(K, device=prop_ids.device)[:num_to_sample]
            mask = torch.zeros(K, dtype=torch.bool, device=prop_ids.device)
            mask[indices] = True
            
        return mask

    # --------------------------- EMA + weights refresh (GPU) ---------------------------

    @torch.no_grad()
    def _update_emas_and_cache_weights(self):
        if (
            self._acc_prop_ids is None
            or self._acc_prop_ids.numel() == 0
            or self._acc_loss_sums is None
            or self._acc_count_sums is None
        ):
            return

        device = self._acc_prop_ids.device
        ema_alpha = self._ema_alpha

        # Per-property mean loss over this accumulation window
        win_means = self._acc_loss_sums / self._acc_count_sums.clamp_min(1.0)  # [K*]

        if self._ema_prop_ids is None:
            # First time initialization
            self._ema_prop_ids = self._acc_prop_ids.clone()
            self._ema_vals = win_means.clone()
            self._prop_counts = self._acc_count_sums.clone()
        else:
            # Union-merge existing EMA set with this window's set
            old_K = self._ema_prop_ids.numel()
            all_props = torch.cat([self._ema_prop_ids, self._acc_prop_ids], dim=0)  # [K_old + K_new]
            uniq, inv = torch.unique(all_props, sorted=True, return_inverse=True)   # uniq: [K], inv: [..]
            K = uniq.numel()

            new_ema_vals = torch.zeros(K, device=device, dtype=self._ema_vals.dtype)
            new_counts  = torch.zeros(K, device=device, dtype=self._prop_counts.dtype)

            inv_old = inv[:old_K]
            new_ema_vals.scatter_add_(0, inv_old, self._ema_vals)
            new_counts.scatter_add_(0, inv_old, self._prop_counts)

            inv_new = inv[old_K:]
            tmp_means  = torch.zeros_like(new_ema_vals)
            tmp_counts = torch.zeros_like(new_counts)

            tmp_means.scatter_add_(0, inv_new, win_means)
            tmp_counts.scatter_add_(0, inv_new, self._acc_count_sums)

            existed = new_counts > 0
            # EMA update for existing; initialize for new
            new_ema_vals[existed] = (1 - ema_alpha) * new_ema_vals[existed] + ema_alpha * tmp_means[existed]
            new_ema_vals[~existed] = tmp_means[~existed]
            new_counts = new_counts + tmp_counts

            self._ema_prop_ids = uniq
            self._ema_vals = new_ema_vals
            self._prop_counts = new_counts

        # Update normalization stats from raw per-token losses collected this window
        if self._acc_stats_vals is not None and self._acc_stats_vals.numel() > 0:
            self._update_train_stats(self._acc_stats_vals)

        # Rebuild weights from EMA vals using current running mean/std
        mean_loss = float(self._train_loss_stats_ema["mean"])
        std_loss  = float(self._train_loss_stats_ema["std"])
        denom = max(std_loss, 1e-6)
        norm = (self._ema_vals - mean_loss) / denom
        norm = norm.clamp(-3, 3)
        t = (norm + 3.0) / 6.0  # in [0,1]
        ratio = self._max_score / self._min_score if self._min_score > 0 else self._max_score
        scores = (self._min_score * (ratio ** t)).clamp(self._min_score, self._max_score)  # [K]

        # NEW: Apply random sampling with bias toward worse properties
        if self._should_apply_random_sampling() and self._ema_vals.numel() > 1:
            sample_mask = self._sample_properties_with_bias(self._ema_prop_ids, self._ema_vals)
            
            # Set non-sampled properties to weight `self.unsampled_property_weight`, keep original weights for sampled ones
            scores[~sample_mask] = self.unsampled_property_weight
            
            # Log statistics about sampling
            num_sampled = sample_mask.sum().item()
            num_total = self._ema_vals.numel()
            if self._batches_processed % 100 == 0:  # Log every 100 batches
                avg_sampled_loss = self._ema_vals[sample_mask].mean().item() if num_sampled > 0 else 0.0
                avg_nonsampled_loss = self._ema_vals[~sample_mask].mean().item() if num_sampled < num_total else 0.0
                logging.info(
                    f"Random sampling: selected {num_sampled}/{num_total} properties "
                    f"(avg_loss: sampled={avg_sampled_loss:.4f}, non-sampled={avg_nonsampled_loss:.4f})"
                )

        # Python dict cache (optional introspection)
        self._cached_score_map = {int(pid): float(w) for pid, w in zip(self._ema_prop_ids.tolist(), scores.tolist())}

        # Keep a sorted tensor view for fast lookups
        self._ema_prop_ids_sorted, order = torch.sort(self._ema_prop_ids)
        self._ema_scores_sorted = scores.index_select(0, order)

        # Clear accumulation buffers
        self._acc_prop_ids = None
        self._acc_loss_sums = None
        self._acc_count_sums = None
        self._acc_stats_vals = None

    # --------------------------- Fast per-batch loss (vectorized) ---------------------------

    def _lookup_weights_for_props(self, uniq_props: torch.Tensor, default_weight: Optional[float] = None) -> torch.Tensor:
        """
        Given uniq_props [K] (long), return weights [K] (float) via searchsorted on cached sorted ids.
        """
        if default_weight is None:
            default_weight = self.unsampled_property_weight  # ← Use configured default
        
        device = uniq_props.device
        if self._ema_prop_ids_sorted is None or self._ema_prop_ids_sorted.numel() == 0:
            return torch.full((uniq_props.numel(),), default_weight, device=device, dtype=torch.float32)

        idx = torch.searchsorted(self._ema_prop_ids_sorted, uniq_props)
        idx_clamped = idx.clamp_max(self._ema_prop_ids_sorted.numel() - 1)
        exact = self._ema_prop_ids_sorted.index_select(0, idx_clamped).eq(uniq_props)
        weights = torch.full((uniq_props.numel(),), default_weight, device=device, dtype=torch.float32)
        if exact.any():
            weights[exact] = self._ema_scores_sorted.index_select(0, idx_clamped[exact])
        return weights

    def _compute_batch_loss(self, logits, values, mask, properties):
        """
        Vectorized per-property CE and weighting.

        IMPORTANT:
        - We group all tokens by property id -> compute *mean CE per property*.
        - Then apply exactly one weight per unique property.
        - Sampled properties get their calculated weights, non-sampled get weight 1.0.
        - If normalize_weights=True, we divide by sum of weights.
        """
        B, T, C = logits.shape
        device = logits.device

        # Flatten
        logits_f = logits.view(-1, C)              # [N, C]
        vals_f   = values.view(-1).to(torch.long)  # [N]
        mask_f   = mask.view(-1).bool()            # [N]
        props_f  = properties.view(-1).to(torch.long)

        if not mask_f.any():
            zero = logits_f.sum() * 0.0
            return zero, torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)

        # Select valid positions
        pos_idx = mask_f.nonzero(as_tuple=False).squeeze(1)  # [M]
        logits_v = logits_f.index_select(0, pos_idx)         # [M, C]
        vals_v   = vals_f.index_select(0, pos_idx)           # [M]
        props_v  = props_f.index_select(0, pos_idx)          # [M]

        # Per-token CE loss (reduction='none')
        ce_per_token = F.cross_entropy(
            logits_v, vals_v, reduction="none", label_smoothing=self.lossfn.label_smoothing
        )  # [M]

        # Map to compact property ids for this batch
        uniq_props, inv = torch.unique(props_v, sorted=True, return_inverse=True)  # uniq_props: [K], inv: [M]
        K = uniq_props.numel()

        # Aggregate sum of losses and counts per property
        loss_sums = torch.zeros(K, device=device, dtype=ce_per_token.dtype)
        cnt_sums  = torch.zeros(K, device=device, dtype=torch.float32)
        loss_sums.scatter_add_(0, inv, ce_per_token)
        cnt_sums.scatter_add_(0, inv, torch.ones_like(inv, dtype=torch.float32))

        per_prop_means = loss_sums / cnt_sums.clamp_min(1.0)  # [K]

        # Cache for EMA update at the *end* of an accumulation window
        is_accum_end = (self.global_step + 1) % self.gradient_accumulation_steps == 0
        if is_accum_end:
            if self._acc_prop_ids is None:
                self._acc_prop_ids  = uniq_props
                self._acc_loss_sums = loss_sums
                self._acc_count_sums= cnt_sums
                self._acc_stats_vals= ce_per_token.detach()
            else:
                self._acc_prop_ids   = torch.cat([self._acc_prop_ids,  uniq_props],  dim=0)
                self._acc_loss_sums  = torch.cat([self._acc_loss_sums, loss_sums],  dim=0)
                self._acc_count_sums = torch.cat([self._acc_count_sums, cnt_sums],  dim=0)
                self._acc_stats_vals = torch.cat([self._acc_stats_vals, ce_per_token.detach()], dim=0)

            # Periodic coalescing to keep buffers small
            self._acc_windows_seen += 1
            if self._acc_windows_seen % self._acc_coalesce_every == 0:
                all_props, inv_all = torch.unique(self._acc_prop_ids, sorted=True, return_inverse=True)
                Kall = all_props.numel()
                new_loss = torch.zeros(Kall, device=device, dtype=self._acc_loss_sums.dtype)
                new_cnt  = torch.zeros(Kall, device=device, dtype=self._acc_count_sums.dtype)
                new_loss.scatter_add_(0, inv_all, self._acc_loss_sums)
                new_cnt.scatter_add_(0, inv_all, self._acc_count_sums)
                self._acc_prop_ids  = all_props
                self._acc_loss_sums = new_loss
                self._acc_count_sums= new_cnt
                # (Keep _acc_stats_vals as raw tokens for running stats)

        # Look up one weight per unique property
        weights_k = self._lookup_weights_for_props(uniq_props, default_weight=1.0).to(per_prop_means.dtype)  # [K]

        # log weights occasionally
        if self._batches_processed % 100 == 0:
            logging.info(
                f"Batch {self._batches_processed}: "
                f"Properties in batch: {K}, "
                f"Weight stats: min={weights_k.min().item():.4f}, max={weights_k.max().item():.4f}, "
                f"mean={weights_k.mean().item():.4f}, std={weights_k.std(unbiased=False).item():.4f}"
            )

        # --- Core objective with random sampling ---
        numerator = (per_prop_means * weights_k).sum()
        if self.normalize_weights:
            denom = weights_k.sum().clamp_min(1e-8)
            weighted_loss = numerator / denom
            batch_weight = denom / max(float(K), 1.0)
        else:
            weighted_loss = numerator
            batch_weight = weights_k.mean()

        raw_loss = per_prop_means.mean()

        return weighted_loss, batch_weight, raw_loss

    # --------------------------- Train one batch ---------------------------

    def _train_batch(self, batch) -> float:
        if self._should_skip_batch():
            skip_reason = f"batch {self._batches_processed + 1}/{self._skip_initial_batches}"
            if self._min_properties_for_weighting is not None:
                skip_reason += f" (properties with EMAs: {self._num_props_with_ema()}/{self._min_properties_for_weighting})"
            logging.info(f"Skipping batch - {skip_reason}")
            self._batches_processed += 1
            return 0.0

        selfies, properties, values, mask = batch

        # Zero grads at the *start* of each accumulation cycle; set_to_none avoids memset
        if self.global_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        selfies    = selfies.to(self.rank, non_blocking=True)
        properties = properties.to(self.rank, non_blocking=True)
        values     = values.to(self.rank, non_blocking=True)
        mask       = mask.to(self.rank, non_blocking=True)

        should_sync = (self.global_step + 1) % self.gradient_accumulation_steps == 0
        context = torch.enable_grad() if should_sync else self.model.no_sync()

        with context:
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                logits = self.model(selfies, properties, values, mask)
                weighted_loss, batch_weight_t, raw_loss_t = self._compute_batch_loss(
                    logits, values, mask, properties
                )
                (weighted_loss / self.gradient_accumulation_steps).backward()
                batch_loss_scalar_t = raw_loss_t.detach()

        if self.log_gpu_mem and (self.global_step % 200 == 0):
            mem_alloc = torch.cuda.memory_allocated() / 1024**3
            mem_rsrv  = torch.cuda.memory_reserved()  / 1024**3
            logging.info(
                f"Step {self.global_step}: GPU memory: {mem_alloc:.2f}GB allocated, {mem_rsrv:.2f}GB reserved"
            )

        if should_sync:
            logging.info("Updating EMAs")
            self._update_emas_and_cache_weights()
            logging.info("EMAs updated")

            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            # Use running mean as the proxy metric for plateau schedulers
            self.step_scheduler(val_metric=self._train_loss_stats_ema["mean"])

        self._batches_processed += 1
        return float(batch_loss_scalar_t.item())