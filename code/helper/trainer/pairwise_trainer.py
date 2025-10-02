# File: pairwise_property_weighted_trainer.py

import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional

from torch.amp import autocast
from torch.nn.parallel import DistributedDataParallel as DDP

# Assuming TrainerCore is available in your environment
from helper.trainer.trainer_core import TrainerCore 

import numpy as np

# ==============================================================================
# Pairwise Ranking Loss Implementation (Integrated)
# ==============================================================================

class PairwiseRankingLoss(nn.Module):
    """
    Implements the Pairwise Ranking Loss (a margin-based hinge loss variant) 
    designed to maximize the Area Under the ROC Curve (AUC) for BINARY classification.

    The per-pair loss L(s_i, s_j) for a positive sample i and a negative sample j is:
        L(s_i, s_j) = max(0, margin - (s_i - s_j)) 
    where s_i and s_j are the predicted scores.
    """

    def __init__(self, margin: float = 1.0, reduction: str = 'mean'):
        super().__init__()
        self.margin = margin
        self.reduction = reduction
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError("Reduction must be 'none', 'mean', or 'sum'.")

    def forward(self, scores: torch.Tensor, labels: torch.Tensor, property_ids: torch.Tensor) -> torch.Tensor:
        """
        Calculates the pairwise ranking loss on a PER-PROPERTY basis.

        Args:
            scores (torch.Tensor): Predicted scores for the positive class (e.g., logits or post-sigmoid values).
                                   Shape: (N,) or (N, 1)
            labels (torch.Tensor): Binary ground truth labels (0 or 1). Shape: (N,)
            property_ids (torch.Tensor): ID for grouping (e.g., user_id or query_id). Pairs are formed 
                                         ONLY within the same property_id. Shape: (N,)
        Returns:
            torch.Tensor: The calculated pairwise ranking loss (sum of all valid pairs).
        """
        
        # --- 1. Pre-processing ---
        scores = scores.view(-1)
        labels = labels.view(-1)
        property_ids = property_ids.view(-1)
        
        if not (scores.shape == labels.shape == property_ids.shape):
            raise ValueError("Scores, Labels, and Property IDs must have the same shape.")

        total_loss = torch.tensor(0.0, device=scores.device, dtype=scores.dtype)
        num_pairs = 0
        
        # --- 2. Per-Property Pairwise Loss (Grouped) ---
        
        for prop_id in torch.unique(property_ids):
            # Isolate samples belonging to the current property
            prop_mask = (property_ids == prop_id)
            prop_scores = scores[prop_mask]
            prop_labels = labels[prop_mask]
            
            # Identify positive and negative samples within this property
            prop_pos_scores = prop_scores[prop_labels == 1]
            prop_neg_scores = prop_scores[prop_labels == 0]
            
            N_pos = prop_pos_scores.numel()
            N_neg = prop_neg_scores.numel()

            if N_pos > 0 and N_neg > 0:
                # [N_pos, 1] - [1, N_neg] = [N_pos, N_neg] matrix of score differences
                diffs = prop_pos_scores.unsqueeze(1) - prop_neg_scores.unsqueeze(0)
                
                # L = max(0, margin - diff) for pairs within this property
                property_pairwise_loss = torch.relu(self.margin - diffs)
                
                total_loss += property_pairwise_loss.sum()
                num_pairs += N_pos * N_neg

        if num_pairs == 0:
            return torch.tensor(0.0, device=scores.device, dtype=scores.dtype)

        # --- 3. Reduction ---
        if self.reduction == 'mean':
            return total_loss / num_pairs
        elif self.reduction == 'sum':
            return total_loss
        else: # 'none'
            return total_loss # Note: returning sum since per-pair matrix is discarded
            
# ==============================================================================
# Pairwise Property Weighted Trainer (Using the new loss)
# ==============================================================================

class PairwisePropertyWeightedTrainer(TrainerCore):
    """
    Trainer that uses PairwiseRankingLoss and computes per-property EMA of losses 
    to derive per-property weights for subsequent batches, optimized for AUC.
    
    NOTE: This is adapted from the Cross-Entropy version to handle the BINARY 
    classification nature of Pairwise Ranking Loss.
    """

    def __init__(
        self,
        *args,
        ema_alpha: float = 0.2,
        min_score: float = 0.1,
        max_score: float = 10.0,
        skip_initial_batches: int = 10,
        min_properties_for_weighting: Optional[int] = None,
        ranking_margin: float = 1.0,  # New parameter for the ranking loss
        log_gpu_mem: bool = False,
        normalize_weights: bool = True,
        acc_coalesce_every: int = 8,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # --- Config
        self._ema_alpha = float(ema_alpha)
        self._min_score = float(min_score)
        self._max_score = float(max_score)

        self._skip_initial_batches = int(skip_initial_batches)
        self._min_properties_for_weighting = min_properties_for_weighting
        self.log_gpu_mem = bool(log_gpu_mem)
        self.normalize_weights = bool(normalize_weights)
        self._acc_coalesce_every = int(acc_coalesce_every)
        self._acc_windows_seen = 0

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
        self._acc_stats_vals: Optional[torch.Tensor] = None # [M*] float (stores per-property mean losses)

        # Loss: Pairwise Ranking Loss (always mean reduction for internal use)
        # We set reduction='sum' internally and manage the total pairs/mean manually
        self.lossfn = PairwiseRankingLoss(margin=ranking_margin, reduction="sum") 

        logging.info(
            "Initialized PairwisePropertyWeightedTrainer with PairwiseRankingLoss "
            f"(ema_alpha={self._ema_alpha}, score_range=[{self._min_score},{self._max_score}], "
            f"skip_initial_batches={self._skip_initial_batches}, ranking_margin={ranking_margin})"
        )

    # --------------------------- Scheduler helpers (Unchanged) ---------------------------

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

    # --------------------------- Weighting gate (Unchanged) ---------------------------

    def _num_props_with_ema(self) -> int:
        return int(self._ema_prop_ids.numel()) if self._ema_prop_ids is not None else 0

    def _should_skip_batch(self) -> bool:
        if self._batches_processed < self._skip_initial_batches:
            return True
        if self._min_properties_for_weighting is not None:
            if self._num_props_with_ema() < self._min_properties_for_weighting:
                return True
        return False

    # --------------------------- Running stats (CPU floats - Adapted) ---------------------------

    def _update_train_stats(self, current_mean_losses: torch.Tensor):
        # NOTE: current_mean_losses here are the K per-property *mean* losses (not per-token)
        if current_mean_losses is None or current_mean_losses.numel() == 0:
            return
        cur_mean = float(current_mean_losses.mean().item())
        # Use a minimum std for stability
        cur_std = float(current_mean_losses.std(unbiased=False).clamp_min(0.1).item())
        
        if not self._train_stats_initialized:
            self._train_loss_stats_ema["mean"] = cur_mean
            self._train_loss_stats_ema["std"] = cur_std
            self._train_stats_initialized = True
        else:
            decay = 0.7
            self._train_loss_stats_ema["mean"] = decay * self._train_loss_stats_ema["mean"] + (1 - decay) * cur_mean
            self._train_loss_stats_ema["std"] = decay * self._train_loss_stats_ema["std"] + (1 - decay) * cur_std

    # --------------------------- EMA + weights refresh (GPU - Adapted) ---------------------------

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
        # For Pairwise Loss, we average the *sum of pairwise losses* L_sum / *total pairs* N_pairs
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
            # Accumulate old EMA values (effectively averaging with the current mean)
            new_ema_vals.scatter_add_(0, inv_old, self._ema_vals)
            new_counts.scatter_add_(0, inv_old, self._prop_counts)

            inv_new = inv[old_K:]
            tmp_means  = torch.zeros_like(new_ema_vals)
            tmp_counts = torch.zeros_like(new_counts)

            # Accumulate new mean losses (per-property mean loss over the window)
            tmp_means.scatter_add_(0, inv_new, win_means)
            tmp_counts.scatter_add_(0, inv_new, self._acc_count_sums)

            # NOTE: Logic is simplified from the CE version since here, the values in
            # tmp_means and new_ema_vals are already *means* (L_sum/N_pairs). 
            # We use a simple decay on the means.

            existed = new_counts > 0
            # EMA update for existing; initialize for new
            new_ema_vals[existed] = (1 - ema_alpha) * new_ema_vals[existed] + ema_alpha * tmp_means[existed]
            new_ema_vals[~existed] = tmp_means[~existed]
            
            # The count is mostly for tracking, but not used in the EMA itself in this structure
            new_counts = new_counts + tmp_counts

            self._ema_prop_ids = uniq
            self._ema_vals = new_ema_vals
            self._prop_counts = new_counts

        # Update normalization stats from raw per-property mean losses collected this window
        if self._acc_stats_vals is not None and self._acc_stats_vals.numel() > 0:
            self._update_train_stats(self._acc_stats_vals) # self._acc_stats_vals holds all per-property means

        # Rebuild weights from EMA vals using current running mean/std
        mean_loss = float(self._train_loss_stats_ema["mean"])
        std_loss  = float(self._train_loss_stats_ema["std"])
        denom = max(std_loss, 1e-6)
        
        # Calculate scores: higher loss -> higher score
        norm = (self._ema_vals - mean_loss) / denom
        norm = norm.clamp(-3, 3)
        t = (norm + 3.0) / 6.0  # in [0,1]
        ratio = self._max_score / self._min_score if self._min_score > 0 else self._max_score
        scores = (self._min_score * (ratio ** t)).clamp(self._min_score, self._max_score)  # [K]

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

    # --------------------------- Fast per-batch loss (vectorized - Unchanged) ---------------------------

    def _lookup_weights_for_props(self, uniq_props: torch.Tensor, default_weight: float = 1.0) -> torch.Tensor:
        """
        Given uniq_props [K] (long), return weights [K] (float) via searchsorted on cached sorted ids.
        """
        device = uniq_props.device
        if self._ema_prop_ids_sorted is None or self._ema_prop_ids_sorted.numel() == 0:
            return torch.full((uniq_props.numel(),), default_weight, device=device, dtype=torch.float32)

        idx = torch.searchsorted(self._ema_prop_ids_sorted, uniq_props)
        # Clamp, then check exact match
        idx_clamped = idx.clamp_max(self._ema_prop_ids_sorted.numel() - 1)
        exact = self._ema_prop_ids_sorted.index_select(0, idx_clamped).eq(uniq_props)
        weights = torch.full((uniq_props.numel(),), default_weight, device=device, dtype=torch.float32)
        if exact.any():
            weights[exact] = self._ema_scores_sorted.index_select(0, idx_clamped[exact])
        return weights

    # --------------------------- Batch Loss Computation (HEAVILY ADAPTED) ---------------------------

    def _compute_batch_loss(self, scores, labels, properties):
        """
        Compute per-property Pairwise Ranking Loss and apply weighting.

        IMPORTANT:
        - The PairwiseRankingLoss is *already* computed per-property internally 
          and returns the SUM of all loss values for that batch.
        - We must recover the K per-property mean losses to update the EMA.
        """
        device = scores.device
        
        # Flatten tensors (assuming scores/labels/properties are already flattened 1D tensors 
        # from the model's output in a typical binary classification setup)
        scores_f = scores.view(-1)
        labels_f = labels.view(-1).to(torch.long)
        props_f  = properties.view(-1).to(torch.long)
        
        if scores_f.numel() == 0:
            return scores.sum() * 0.0, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)

        # --- 1. Compute Loss and Identify Per-Property Losses/Counts for EMA ---
        
        uniq_props = torch.unique(props_f)
        K = uniq_props.numel()

        loss_sums = torch.zeros(K, device=device, dtype=scores_f.dtype)
        cnt_pairs = torch.zeros(K, device=device, dtype=torch.float32)

        # Total unweighted loss sum over all pairs in the batch
        total_unweighted_sum_loss = torch.tensor(0.0, device=device, dtype=scores_f.dtype)
        total_num_pairs = 0

        # This loop is necessary to extract per-property loss/pair count for EMA
        # The PairwiseRankingLoss *only* calculates the overall sum/mean by default.
        for i, prop_id in enumerate(uniq_props):
            prop_mask = (props_f == prop_id)
            prop_scores = scores_f[prop_mask]
            prop_labels = labels_f[prop_mask]
            
            pos_scores = prop_scores[prop_labels == 1]
            neg_scores = prop_scores[prop_labels == 0]
            N_pos = pos_scores.numel()
            N_neg = neg_scores.numel()
            
            if N_pos > 0 and N_neg > 0:
                diffs = pos_scores.unsqueeze(1) - neg_scores.unsqueeze(0)
                # L_sum_p: Sum of all pairwise losses for this property
                L_sum_p = torch.relu(self.lossfn.margin - diffs).sum() 
                N_pairs_p = N_pos * N_neg
                
                loss_sums[i] = L_sum_p
                cnt_pairs[i] = N_pairs_p
                
                total_unweighted_sum_loss += L_sum_p
                total_num_pairs += N_pairs_p
        
        if total_num_pairs == 0:
             zero = scores.sum() * 0.0
             return zero, torch.tensor(1.0, device=device), torch.tensor(0.0, device=device)

        # Per-property *mean* loss (L_sum / N_pairs)
        per_prop_means = loss_sums / cnt_pairs.clamp_min(1.0) # [K]
        raw_loss = total_unweighted_sum_loss / total_num_pairs # Scalar mean loss for logging

        # --- 2. Cache for EMA update at the *end* of an accumulation window ---
        
        is_accum_end = (self.global_step + 1) % self.gradient_accumulation_steps == 0
        if is_accum_end:
            # We store the unique IDs, sum of losses, count of pairs, and the per-prop means 
            # (the mean loss is used to update the running statistics)
            if self._acc_prop_ids is None:
                self._acc_prop_ids  = uniq_props
                self._acc_loss_sums = loss_sums
                self._acc_count_sums= cnt_pairs
                self._acc_stats_vals= per_prop_means.detach()
            else:
                self._acc_prop_ids   = torch.cat([self._acc_prop_ids,  uniq_props],  dim=0)
                self._acc_loss_sums  = torch.cat([self._acc_loss_sums, loss_sums],  dim=0)
                self._acc_count_sums = torch.cat([self._acc_count_sums, cnt_pairs],  dim=0)
                self._acc_stats_vals = torch.cat([self._acc_stats_vals, per_prop_means.detach()], dim=0)

            # Periodic coalescing to keep buffers small
            self._acc_windows_seen += 1
            if self._acc_windows_seen % self._acc_coalesce_every == 0:
                all_props, inv_all = torch.unique(self._acc_prop_ids, sorted=True, return_inverse=True)
                Kall = all_props.numel()
                new_loss = torch.zeros(Kall, device=device, dtype=self._acc_loss_sums.dtype)
                new_cnt  = torch.zeros(Kall, device=device, dtype=self._acc_count_sums.dtype)
                
                # Coalesce the sums and counts
                new_loss.scatter_add_(0, inv_all, self._acc_loss_sums)
                new_cnt.scatter_add_(0, inv_all, self._acc_count_sums)
                
                self._acc_prop_ids  = all_props
                self._acc_loss_sums = new_loss
                self._acc_count_sums= new_cnt
                # (Keep _acc_stats_vals as raw tokens for running stats)


        # --- 3. Apply Weighting ---

        # Look up one weight per unique property
        weights_k = self._lookup_weights_for_props(uniq_props, default_weight=1.0).to(per_prop_means.dtype)  # [K]

        # Core objective: weight_p * mean_loss_p (L_sum_p / N_pairs_p)
        weighted_loss_sum = (per_prop_means * weights_k).sum()
        
        if self.normalize_weights:
            # Normalize by sum of weights (for scale stability)
            denom = weights_k.sum().clamp_min(1e-8)
            weighted_loss = weighted_loss_sum / denom
            batch_weight = denom / max(float(K), 1.0)  # logging proxy (avg weight scale)
        else:
            # Simple sum of weighted mean losses
            weighted_loss = weighted_loss_sum
            batch_weight = weights_k.mean()

        # The weighted_loss represents the sum of (weighted mean losses).
        return weighted_loss, batch_weight, raw_loss

    # --------------------------- Train one batch (Adapted) ---------------------------

    def _train_batch(self, batch) -> float:
        if self._should_skip_batch():
            skip_reason = f"batch {self._batches_processed + 1}/{self._skip_initial_batches}"
            if self._min_properties_for_weighting is not None:
                skip_reason += f" (properties with EMAs: {self._num_props_with_ema()}/{self._min_properties_for_weighting})"
            logging.info(f"Skipping batch - {skip_reason}")
            self._batches_processed += 1
            return 0.0

        # Assuming batch now contains only scores, properties, and binary labels for the ranking task
        scores, properties, labels = batch # NOTE: Changed batch structure assumption from CE version

        # Zero grads at the *start* of each accumulation cycle; set_to_none avoids memset
        if self.global_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad(set_to_none=True)

        scores     = scores.to(self.rank, non_blocking=True)
        properties = properties.to(self.rank, non_blocking=True)
        labels     = labels.to(self.rank, non_blocking=True)

        should_sync = (self.global_step + 1) % self.gradient_accumulation_steps == 0
        context = torch.enable_grad() if should_sync else self.model.no_sync()

        with context:
            # Assuming model output is a single score/logit for binary classification
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=True):
                # NOTE: The model call needs to be adapted for your specific model's input
                model_scores = self.model(scores, properties) # Generic call, adjust as needed

                weighted_loss, batch_weight_t, raw_loss_t = self._compute_batch_loss(
                    model_scores.view(-1), labels.view(-1), properties.view(-1)
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