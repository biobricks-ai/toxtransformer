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

class TrainerCore(nn.Module):
    """
    Abstract base: handles device/DDP, accumulation, scheduler, metrics, eval, and the main loop.
    Subclasses implement `_train_batch(self, batch) -> float`.
    """

    def __init__(self,
                 model: nn.Module,
                 rank: int,
                 tokenizer,
                 trn_iterator,
                 batch_size: int,
                 scheduler,                       # must expose `.optimizer`
                 max_steps: int = 100_000,
                 first_eval: int = 100,
                 eval_every: int = 10_000,
                 eval_samples: int = 400,
                 effective_accum_batch_size: int = 1024,
                 find_unused_parameters: bool = False):
        super().__init__()
        self.rank = rank
        self.global_step = 0
        self.trn_iterator = trn_iterator
        self.tokenizer = tokenizer

        torch.cuda.set_device(rank)
        self.model = model.to(rank)

        if dist.is_initialized():
            dist.barrier(device_ids=[self.rank])
            self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=find_unused_parameters)

        self.max_steps = max_steps
        self.first_eval = first_eval
        self.eval_every = eval_every
        self.eval_samples = eval_samples

        world = dist.get_world_size() if dist.is_initialized() else 1
        self.gradient_accumulation_steps = max(1, effective_accum_batch_size // (batch_size * world))

        self.scheduler = scheduler
        self.optimizer = scheduler.optimizer

        self.metrics_path = None
        self.best_loss = np.inf
        self.best_auc = 0.0
        self.valdl = None

        # For tracking moving averages
        self.loss_history = []
        self.loss_window = 100  # Track last 100 steps for moving average

        logging.info(
            f"TrainerCore: rank={rank}, grad_accum={self.gradient_accumulation_steps}, "
            f"world={world}, max_steps={max_steps}"
        )

    # --- hooks ---

    def _train_batch(self, batch) -> float:
        """Subclasses must implement. Must return *de-accumulated* scalar loss for logging."""
        raise NotImplementedError

    # --- utilities / shared ---

    def set_model_savepath(self, savepath: str):
        self.savepath = pathlib.Path(savepath)
        self.savepath.mkdir(exist_ok=True, parents=True)
        if self.rank == 0:
            logging.info(f"Model save path: {self.savepath}")
        return self

    def set_validation_dataloader(self, valdl):
        self.valdl = valdl
        return self

    def set_metrics_file(self, metrics_path: str, overwrite: bool = False):
        self.metrics_path = pathlib.Path(metrics_path)
        if self.rank == 0:
            self.metrics_path.parent.mkdir(exist_ok=True, parents=True)
            if overwrite:
                with open(self.metrics_path, 'w') as f:
                    f.write("type\tbatch\tloss\tlr\tauc\tbac\n")
        return self

    @torch.no_grad()
    def _eval_all(self, max_eval_batches: Optional[int] = None) -> Dict[str, float]:
        assert self.valdl is not None, "Validation dataloader not set"
        torch.cuda.empty_cache()
        max_eval_batches = self.eval_samples if max_eval_batches is None else max_eval_batches
        max_eval_batches = min(max_eval_batches, len(self.valdl))

        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        all_preds, all_targets = [], []

        if dist.is_initialized():
            dist.barrier(device_ids=[self.rank])

        for i, (selfies, properties, values, mask) in enumerate(self.valdl):
            if i >= max_eval_batches:
                break

            selfies = selfies.to(self.rank, non_blocking=True)
            properties = properties.to(self.rank, non_blocking=True)
            values = values.to(self.rank, non_blocking=True)
            mask = mask.to(self.rank, non_blocking=True)

            with autocast(device_type='cuda', dtype=torch.bfloat16):
                logits = self.model(selfies, properties, values, mask)

                # masked CE
                B, T, C = logits.shape
                logits_f = logits.view(-1, C)
                vals_f   = values.view(-1)
                mask_f   = mask.view(-1)
                if mask_f.any():
                    loss = nn.functional.cross_entropy(logits_f[mask_f], vals_f[mask_f])
                else:
                    loss = logits_f.new_zeros(())

                probs = F.softmax(logits, dim=-1)
                probs_f = probs.view(-1, C)

                if mask_f.any():
                    all_preds.append(probs_f[mask_f])
                    all_targets.append(vals_f[mask_f])

                total_loss += float(loss)
                num_batches += 1

        # Gather loss across ranks
        mean_loss = total_loss / max(1, num_batches)
        if dist.is_initialized():
            loss_t = torch.tensor(total_loss, device=self.rank)  # ✅ Sum raw losses
            nb_t = torch.tensor(num_batches, device=self.rank)
            dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
            dist.all_reduce(nb_t, op=dist.ReduceOp.SUM)
            mean_loss = loss_t.item() / max(1, nb_t.item())
        else:
            mean_loss = total_loss / max(1, num_batches)

        # Metrics
        if all_preds:
           ap = torch.cat(all_preds, dim=0)
           at = torch.cat(all_targets, dim=0)

           if dist.is_initialized():
               local_len = torch.tensor([ap.size(0)], device=self.rank)
               lengths = [torch.zeros_like(local_len) for _ in range(dist.get_world_size())]
               dist.all_gather(lengths, local_len)
               
               lengths_list = [l.item() for l in lengths]
               max_len = max(lengths_list)
               
               if self.rank == 0:
                   logging.info(f"Gather: {lengths_list} samples per rank, total {sum(lengths_list)}")
               
               if max_len > ap.size(0):
                   ap = F.pad(ap, (0, 0, 0, max_len - ap.size(0)))
                   at = F.pad(at, (0, max_len - at.size(0)), value=-1)

               g_preds = [torch.zeros_like(ap) for _ in range(dist.get_world_size())]
               g_targs = [torch.zeros_like(at) for _ in range(dist.get_world_size())]
               dist.all_gather(g_preds, ap)
               dist.all_gather(g_targs, at)

               # Only take valid samples from each rank
               ap_list = [g_preds[i][:lengths_list[i]] for i in range(len(lengths_list)) if lengths_list[i] > 0]
               at_list = [g_targs[i][:lengths_list[i]] for i in range(len(lengths_list)) if lengths_list[i] > 0]
               
               ap = torch.cat(ap_list, dim=0).float().cpu().numpy() if ap_list else np.array([]).reshape(0, ap.size(1))
               at = torch.cat(at_list, dim=0).cpu().numpy() if at_list else np.array([])
           else:
               ap = ap.float().cpu().numpy()
               at = at.cpu().numpy()

           valid = at >= 0
           ap = ap[valid]
           at = at[valid]
           
           if self.rank == 0:
               logging.info(f"Metrics on {len(at)} samples, targets: {np.unique(at)}")

           unique_targets = np.unique(at)
           if len(at) > 0 and len(unique_targets) == 2 and ap.shape[1] >= 2:
               try:
                   auc = roc_auc_score(at, ap[:, 1])
                   bac = balanced_accuracy_score(at, ap.argmax(axis=1))
               except Exception as e:
                   if self.rank == 0:
                       logging.warning(f"Metric error: {e}")
                   auc = 0.0
                   bac = 0.0
           else:
               auc, bac = 0.0, 0.0

        logging.info(f"📊 EVAL METRICS [Step {self.global_step}] - Loss: {mean_loss:.4f}, AUC: {auc:.4f}, BAC: {bac:.4f}")
        return {"loss": mean_loss, "auc": auc, "bac": bac}

    def start(self):
        logging.info(f"🚀 Starting training loop - Max steps: {self.max_steps}, First eval: {self.first_eval}, Eval every: {self.eval_every}")
        
        epoch = 0
        while self.global_step < self.max_steps:
            if hasattr(self.trn_iterator, "sampler") and hasattr(self.trn_iterator.sampler, "set_epoch"):
                self.trn_iterator.sampler.set_epoch(epoch)

            if dist.is_initialized():
                dist.barrier(device_ids=[self.rank])
            self.model.train()

            for i, batch in enumerate(self.trn_iterator):
                loss = self._train_batch(batch)
                logging.info(f"Step {self.global_step}: Loss: {loss:.4f}")

                # Track loss history for moving average
                self.loss_history.append(loss)
                if len(self.loss_history) > self.loss_window:
                    self.loss_history.pop(0)
                
                current_lr = self.optimizer.param_groups[0]['lr']
                
                # Log training progress every accumulation step
                if self.global_step % self.gradient_accumulation_steps == 0 and self.rank == 0:
                    avg_loss = sum(self.loss_history) / len(self.loss_history)
                    progress = (self.global_step / self.max_steps) * 100
                    
                    logging.info(
                        f"🔥 TRAIN [Step {self.global_step:>6}/{self.max_steps}] ({progress:5.1f}%) - "
                        f"Loss: {loss:.4f} (avg: {avg_loss:.4f}), LR: {current_lr:.2e}"
                    )
                    
                    if self.metrics_path:
                        with open(self.metrics_path, 'a') as f:
                            f.write(f"train\t{self.global_step}\t{loss:.4f}\t{current_lr:.6f}\t0.0\t0.0\n")

                # evaluation schedule
                if self.global_step == self.first_eval or (self.global_step + self.first_eval) % self.eval_every == 0:
                    logging.info(f"🔍 Starting evaluation at step {self.global_step}...")

                    if dist.is_initialized():
                        torch.cuda.synchronize(self.rank)
                        dist.barrier(device_ids=[self.rank])

                    self.model.eval()
                    with torch.no_grad():
                        evals = self._eval_all(max_eval_batches=self.eval_samples)
                    
                    if self.rank == 0:
                        # Check for improvements
                        is_best_loss = evals['loss'] < self.best_loss
                        is_best_auc = evals['auc'] > self.best_auc
                        
                        if is_best_loss:
                            self.best_loss = evals['loss']
                        if is_best_auc:
                            self.best_auc = evals['auc']
                        
                        improvement_str = ""
                        if is_best_loss:
                            improvement_str += " 🎯 BEST LOSS!"
                        if is_best_auc:
                            improvement_str += " 🎯 BEST AUC!"
                        
                        logging.info(
                            f"✅ EVAL COMPLETE [Step {self.global_step}] - "
                            f"Loss: {evals['loss']:.4f} (best: {self.best_loss:.4f}), "
                            f"AUC: {evals['auc']:.4f} (best: {self.best_auc:.4f}), "
                            f"BAC: {evals['bac']:.4f}, LR: {current_lr:.2e}{improvement_str}"
                        )
                        
                        if self.metrics_path:
                            with open(self.metrics_path, 'a') as f:
                                f.write(f"eval\t{self.global_step}\t{evals['loss']:.4f}\t{current_lr:.6f}\t{evals['auc']:.4f}\t{evals['bac']:.4f}\n")

                    if dist.is_initialized():
                        dist.barrier(device_ids=[self.rank])
                    self.model.train()
                
                self.global_step += 1

            epoch += 1
            logging.info(f"📅 Completed epoch {epoch}")

        logging.info(
            f"🏁 Training finished at step {self.global_step}! "
                f"Best loss: {self.best_loss:.4f}, Best AUC: {self.best_auc:.4f}"
            )
