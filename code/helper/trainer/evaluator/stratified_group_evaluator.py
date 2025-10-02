import logging
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import sqlite3
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.amp import autocast
from sklearn.metrics import roc_auc_score


class StratifiedGroupEvaluator:
    """
    DDP-optimized evaluator:
      • per-rank accumulation of scores/targets per property
      • local per-property AUC on each rank
      • communicate only tensors of shape [num_tasks] for (auc*n) and n
      • losses communicated as (sum, count) tensors
      • group AUC = mean(property AUCs in group with data)
      • stratified AUC = mean(group AUCs)
      • group loss = mean over properties (mean loss per property)
      • stratified loss = mean over groups of group loss
    """

    def __init__(self, tokenizer, rank: int, label_smoothing: float = 0.05):
        self.rank = rank
        self.device = f'cuda:{rank}' if torch.cuda.is_available() else 'cpu'  # FIXED
        self._groups: Dict[str, List[int]] = self._build_property_groups(tokenizer)
        self._group_names = list(self._groups.keys())
        self._num_tasks = max([p for props in self._groups.values() for p in props]) + 1
        self.lossfn = nn.CrossEntropyLoss(reduction="mean", label_smoothing=label_smoothing)

        logging.info(f"[rank {self.rank}] StratifiedGroupEvaluator: {len(self._group_names)} groups: {self._group_names}")
        logging.info(f"[rank {self.rank}] Device: {self.device}, Num tasks: {self._num_tasks}")

    # ---------- grouping ----------
    def _build_property_groups(self, tokenizer) -> Dict[str, List[int]]:
        bp = pd.read_parquet('cache/get_benchmark_properties/benchmark_properties.parquet')
        conn = sqlite3.connect('brick/cvae.sqlite')
        prop_src = pd.read_sql(
            'SELECT property_token,title,source FROM property p INNER JOIN source s on p.source_id = s.source_id',
            conn
        )
        conn.close()

        merged = bp.merge(prop_src, on=['property_token', 'source'], how='inner')
        merged = merged.query('source != "ctdbase"')  # ctdbase is noisy / low-quality
        groups = {}
        for source, gdf in merged.groupby('source'):
            props = (np.array(gdf['property_token'].unique().tolist()) - tokenizer.selfies_offset).tolist()
            groups[source] = sorted([p for p in props if p >= 0])
        return groups

    # ---------- per-batch accumulation ----------
    @torch.no_grad()
    def _collect_batch(self, logits, properties, values, mask):
        """
        Returns (local, no DDP yet):
          scores_by_prop : dict[int, list[float]]  (prob of class-1)
          targets_by_prop: dict[int, list[int]]    (0/1)
          loss_sum       : 1D tensor [num_tasks]
          count          : 1D tensor [num_tasks]
        """
        B, T, C = logits.shape
        logits = logits.reshape(-1, C)
        props  = properties.reshape(-1)
        targs  = values.reshape(-1)
        m      = mask.reshape(-1)

        if not torch.any(m):
            zf = torch.zeros(self._num_tasks, device=self.device)  # FIXED
            zi = torch.zeros(self._num_tasks, device=self.device, dtype=torch.int64)  # FIXED
            return {}, {}, zf, zi

        logits = logits[m]
        props  = props[m]
        targs  = targs[m]

        valid = (props >= 0) & (props < self._num_tasks)
        if not torch.any(valid):
            # ADDED: Log when we have invalid properties for debugging
            invalid_count = (~valid).sum().item()
            if invalid_count > 0:
                logging.debug(f"[rank {self.rank}] Found {invalid_count} invalid property indices")
            
            zf = torch.zeros(self._num_tasks, device=self.device)  # FIXED
            zi = torch.zeros(self._num_tasks, device=self.device, dtype=torch.int64)  # FIXED
            return {}, {}, zf, zi

        logits = logits[valid]
        props  = props[valid].long()
        targs  = targs[valid].long()

        # sample-wise loss for per-property mean
        losses = F.cross_entropy(
            logits, targs, reduction="none",
            label_smoothing=self.lossfn.label_smoothing
        )

        loss_sum = torch.zeros(self._num_tasks, device=self.device, dtype=losses.dtype)  # FIXED
        count    = torch.zeros(self._num_tasks, device=self.device, dtype=torch.int64)  # FIXED

        loss_sum.scatter_add_(0, props, losses)
        count.scatter_add_(0, props, torch.ones_like(props, dtype=torch.int64))

        # OPTIMIZED: Keep data on GPU longer, move to CPU only what's needed
        probs = F.softmax(logits, dim=-1)[:, 1].detach().float()
        ys    = targs.detach().int()

        scores_by_prop: Dict[int, List[float]] = {}
        targets_by_prop: Dict[int, List[int]] = {}
        for p in torch.unique(props).tolist():
            sel = (props == p)
            # Move to CPU only the selected slices
            scores_by_prop[int(p)]  = probs[sel].cpu().tolist()
            targets_by_prop[int(p)] = ys[sel].cpu().tolist()

        return scores_by_prop, targets_by_prop, loss_sum, count

    # ---------- metric math (local) ---------- 
    def _local_property_aucs(self, scores_by_prop: Dict[int, List[float]], targets_by_prop: Dict[int, List[int]]):
        """
        Compute AUC per property *on this rank only*.
        Returns:
          auc_vec   : tensor [num_tasks] with local auc (0 if not computable)
          auc_count : tensor [num_tasks] with number of local samples used for that AUC
        """
        auc_vec   = torch.zeros(self._num_tasks, device=self.device, dtype=torch.float32)  # FIXED
        auc_count = torch.zeros(self._num_tasks, device=self.device, dtype=torch.int64)   # FIXED

        for p, scores in scores_by_prop.items():
            ys = targets_by_prop.get(p, [])
            if not scores or not ys:
                continue
            # need both classes locally to get a meaningful AUC
            if len(set(ys)) < 2:
                logging.debug(f"[rank {self.rank}] Property {p} has only one class locally, skipping AUC")
                continue
            try:
                auc = roc_auc_score(ys, scores)
                auc_vec[p]   = float(auc)
                auc_count[p] = len(ys)
                logging.debug(f"[rank {self.rank}] local AUC prop {p}: {auc:.4f} (n={len(ys)})")
            except Exception as e:
                logging.warning(f"[rank {self.rank}] AUC error for property {p}: {e}")
        return auc_vec, auc_count

    # ---------- IMPROVED: Better AUC aggregation ----------
    def _compute_global_auc_improved(self, local_scores_by_prop: Dict[int, List[float]], 
                                   local_targets_by_prop: Dict[int, List[int]]) -> Dict[int, float]:
        """
        IMPROVED: Gather all scores/targets globally and compute true global AUC.
        This is more accurate than weighted averaging of per-rank AUCs.
        """
        world_size = dist.get_world_size()
        prop_aucs: Dict[int, float] = {}
        
        # For each property that appears locally, gather data from all ranks
        all_local_props = set(local_scores_by_prop.keys())
        
        # Gather which properties each rank has
        local_props_tensor = torch.zeros(self._num_tasks, device=self.device, dtype=torch.bool)
        for p in all_local_props:
            local_props_tensor[p] = True
        
        global_props_tensor = torch.zeros_like(local_props_tensor)
        dist.all_reduce(global_props_tensor, op=dist.ReduceOp.MAX)  # OR operation
        
        global_props = torch.nonzero(global_props_tensor, as_tuple=True)[0].tolist()
        
        for prop in global_props:
            try:
                # Gather data sizes first
                local_size = len(local_scores_by_prop.get(prop, []))
                sizes = [torch.tensor(0, device=self.device) for _ in range(world_size)]
                dist.all_gather(sizes, torch.tensor(local_size, device=self.device))
                sizes = [s.item() for s in sizes]
                
                if sum(sizes) == 0:
                    continue
                    
                max_size = max(sizes)
                if max_size == 0:
                    continue
                
                # Prepare padded tensors
                local_scores = local_scores_by_prop.get(prop, [])
                local_targets = local_targets_by_prop.get(prop, [])
                
                padded_scores = torch.zeros(max_size, device=self.device)
                padded_targets = torch.zeros(max_size, device=self.device, dtype=torch.long)
                
                if len(local_scores) > 0:
                    padded_scores[:len(local_scores)] = torch.tensor(local_scores, device=self.device)
                    padded_targets[:len(local_targets)] = torch.tensor(local_targets, device=self.device)
                
                # Gather from all ranks
                gathered_scores = [torch.zeros(max_size, device=self.device) for _ in range(world_size)]
                gathered_targets = [torch.zeros(max_size, device=self.device, dtype=torch.long) for _ in range(world_size)]
                
                dist.all_gather(gathered_scores, padded_scores)
                dist.all_gather(gathered_targets, padded_targets)
                
                # Concatenate valid data
                all_scores = []
                all_targets = []
                for rank_idx, (scores_t, targets_t) in enumerate(zip(gathered_scores, gathered_targets)):
                    valid_size = sizes[rank_idx]
                    if valid_size > 0:
                        all_scores.extend(scores_t[:valid_size].cpu().tolist())
                        all_targets.extend(targets_t[:valid_size].cpu().tolist())
                
                # Compute global AUC
                if len(set(all_targets)) >= 2 and len(all_scores) > 0:
                    global_auc = roc_auc_score(all_targets, all_scores)
                    prop_aucs[prop] = float(global_auc)
                    if self.rank == 0:  # Log only from rank 0
                        logging.debug(f"Global AUC prop {prop}: {global_auc:.4f} (n={len(all_scores)})")
                        
            except Exception as e:
                logging.warning(f"[rank {self.rank}] Global AUC error for property {prop}: {e}")
                
        return prop_aucs

    # ---------- public API ----------
    @torch.no_grad()
    def evaluate(self, model: nn.Module, valdl, max_eval_batches: Optional[int] = None, 
                 use_improved_auc: bool = False) -> Dict[str, float]:
        model.eval()
        max_eval_batches = min(max_eval_batches or len(valdl), len(valdl))
        
        dist.barrier()  # sync before evaluation

        # local accumulators
        local_scores_by_prop: Dict[int, List[float]] = {}
        local_targets_by_prop: Dict[int, List[int]] = {}
        loss_sum = torch.zeros(self._num_tasks, device=self.device)  # FIXED
        count    = torch.zeros(self._num_tasks, device=self.device, dtype=torch.int64)  # FIXED

        for i, (selfies, properties, values, mask) in enumerate(valdl):
            if i >= max_eval_batches:
                break

            selfies    = selfies.to(self.device, non_blocking=True)  # FIXED: use self.device
            properties = properties.to(self.device, non_blocking=True)
            values     = values.to(self.device, non_blocking=True)
            mask       = mask.to(self.device, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = model(selfies, properties, values, mask)

            # sbp: scores_by_prop - Dict[int, List[float]] - predicted probabilities (class 1) grouped by property ID
            # tbp: targets_by_prop - Dict[int, List[int]] - true binary labels (0/1) grouped by property ID  
            # lsum: loss_sum - Tensor[num_tasks] - sum of cross-entropy losses per property across batch
            # cnt: count - Tensor[num_tasks] - number of valid samples per property in this batch
            sbp, tbp, lsum, cnt = self._collect_batch(logits, properties, values, mask)

            # merge local lists (no inter-rank comms)
            for p, lst in sbp.items():
                local_scores_by_prop.setdefault(p, []).extend(lst)
            for p, lst in tbp.items():
                local_targets_by_prop.setdefault(p, []).extend(lst)

            loss_sum += lsum
            count    += cnt

        # ADDED: Error handling for reductions
        try:
            # reduce losses exactly
            dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
            dist.all_reduce(count,    op=dist.ReduceOp.SUM)
        except Exception as e:
            logging.error(f"[rank {self.rank}] Loss reduction failed: {e}")
            raise

        # Choose AUC computation method
        if use_improved_auc:
            prop_aucs = self._compute_global_auc_improved(local_scores_by_prop, local_targets_by_prop)
        else:
            # Original method (faster but less accurate)
            # local AUCs (no comms yet)
            local_auc_vec, local_auc_count = self._local_property_aucs(local_scores_by_prop, local_targets_by_prop)

            # communicate only (auc*n) and n — count-weighted average of per-rank AUCs
            auc_weighted_sum = local_auc_vec * local_auc_count.clamp_min(0).to(local_auc_vec.dtype)
            
            try:
                dist.all_reduce(auc_weighted_sum, op=dist.ReduceOp.SUM)
                dist.all_reduce(local_auc_count,  op=dist.ReduceOp.SUM)
            except Exception as e:
                logging.error(f"[rank {self.rank}] AUC reduction failed: {e}")
                raise

            # finalize global per-property AUCs (approximate)
            prop_aucs: Dict[int, float] = {}
            for p in range(self._num_tasks):
                n = int(local_auc_count[p].item())
                if n > 0:
                    prop_aucs[p] = float((auc_weighted_sum[p] / n).item())

        # per-property mean loss from (sum, count)
        prop_mean_loss = torch.zeros_like(loss_sum, dtype=torch.float32)
        nz = count > 0
        prop_mean_loss[nz] = (loss_sum[nz] / count[nz]).float()

        # group metrics
        group_losses: Dict[str, float] = {}
        for gname, props in self._groups.items():
            idx = [p for p in props if p < self._num_tasks and count[p].item() > 0]
            if idx:
                group_losses[gname] = float(prop_mean_loss[idx].mean().item())
            else:
                group_losses[gname] = 0.0

        group_aucs = {}
        for gname, props in self._groups.items():
            vals = [prop_aucs[p] for p in props if p in prop_aucs]
            group_aucs[gname] = float(np.mean(vals)) if len(vals) else 0.0

        # stratified (means over groups)
        stratified_auc  = float(np.mean(list(group_aucs.values()))) if group_aucs else 0.0
        stratified_loss = float(np.mean(list(group_losses.values()))) if group_losses else 0.0

        # pack + log
        result = {
            "loss": stratified_loss,
            "auc": stratified_auc,
            "auc_stratified": stratified_auc,
            "loss_stratified": stratified_loss,
        }
        for g in self._group_names:
            result[f"auc_group_{g}"] = group_aucs.get(g, 0.0)
            result[f"loss_group_{g}"] = group_losses.get(g, 0.0)

        # Log count of occurrences for each group
        for gname, props in self._groups.items():
            group_count = sum(count[p].item() for p in props if p < self._num_tasks)
            result[f"count_group_{gname}"] = group_count

        if self.rank == 0:  # Log only from rank 0 to avoid spam
            logging.info(f"Stratified AUC: {result['auc_stratified']:.4f}")
            logging.info(f"Stratified Loss: {result['loss_stratified']:.4f}")
            for g in self._group_names:
                logging.info(f"[{g:>15}] AUC={result[f'auc_group_{g}']:.4f} Loss={result[f'loss_group_{g}']:.4f} CNT={result[f'count_group_{g}']:>8}")

        return result