from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
# from lion_pytorch import Lion
import logging
import pathlib
import numpy as np
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import traceback # Import traceback to log full error info
import psutil
import datetime
from cvae.models.multitask_transformer import linear_warmup_and_decay_scheduler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR, CosineAnnealingWarmRestarts
from typing import Optional, Tuple

class Trainer():

    def __init__(self, model, rank, tokenizer, trn_iterator, batch_size, first_eval=100, eval_every=10000, 
    eval_samples=400, scheduler_warmup_steps=10000, scheduler_max_steps=100000, max_steps=100000,
    scheduler_min_lr=1e-6, scheduler_max_lr=3e-4, effective_accum_batch_size=1024, find_unused_parameters=False):

        self.rank = rank
        self.global_step = 0
        self.trn_iterator = trn_iterator
        self.tokenizer = tokenizer

        torch.cuda.set_device(rank)
        logging.info(f"Rank {self.rank}: Setting CUDA device to {rank}")
        self.model = model.to(rank)
        self.lossfn = self.model.build_stratified_lossfn()
        self.eval_loss = self.model.build_stratified_lossfn()
        
        dist.barrier(device_ids=[self.rank]) # Explicitly passing device_ids
        # self.model = torch.compile(self.model)
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=find_unused_parameters)
        self.max_steps = max_steps # Total training steps defined here

        # Add gradient accumulation for effectively larger batch sizes
        self.gradient_accumulation_steps = max(1, effective_accum_batch_size // (batch_size * dist.get_world_size()))
        
        # Calculate effective batch size per GPU
        effective_batch_size_per_gpu = batch_size * self.gradient_accumulation_steps
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        total_effective_batch_size = effective_batch_size_per_gpu * world_size # Use effective_batch_size_per_gpu


        # --- LR and Scheduler Setup (UPDATED) ---
        # Define learning rate parameters clearly
        peak_lr = scheduler_max_lr # Your desired maximum LR (e.g., 3e-4 or 1e-2 from user input)
        min_lr = scheduler_min_lr # Your desired minimum LR (e.g., 1e-6 or 1e-5 from user input)
        warmup_total_iters = scheduler_warmup_steps # Number of steps for linear warmup
        # T_0 for cosine annealing, represents the length of the first cosine cycle.
        # It's usually a significant portion of total training steps.
        cosine_cycle_len = scheduler_max_steps 

        # Optimizer is initialized with the 'start' LR for the SequentialLR (which is min_lr for warmup)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=peak_lr, betas=(0.9, 0.99), weight_decay=1e-2)

        # 1. Warm-up scheduler: Linear increase from min_lr to peak_lr
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=min_lr / peak_lr, # Factor to scale optimizer's initial LR (peak_lr) down to min_lr
            end_factor=1.0,                # Factor to bring optimizer's initial LR (peak_lr) back to peak_lr
            total_iters=warmup_total_iters
        )
        
        # 2. Main Cosine Annealing scheduler (starts after warm-up)
        cosine_decay_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=cosine_cycle_len, # This sets the duration of the first cosine cycle
            T_mult=2,             # Each restart cycle is 2x longer
            eta_min=min_lr        # Min learning rate at end of each cycle
        )

        # Combine them: Warm-up first, then Cosine Annealing
        self.scheduler = SequentialLR(
            self.optimizer, 
            schedulers=[warmup_scheduler, cosine_decay_scheduler], 
            milestones=[warmup_total_iters] # Switch from warmup_scheduler to cosine_decay_scheduler after warmup_total_iters
        )
        # --- END LR and Scheduler Setup ---


        self.metrics_path = None
        self.best_loss = np.inf

        # Update GradScaler initialization with device parameter
        self.scaler = GradScaler()

        # Reduce evaluation frequency but evaluate quickly to trigger model saving
        self.first_eval = first_eval
        self.eval_every = eval_every
        self.eval_samples = eval_samples # Number of batches to use for evaluation

    
        logging.info(f"""Trainer initialized with:
        - Model: {self.model.__class__.__name__}
        - Rank: {self.rank}
        - Training samples: {len(self.trn_iterator.dataset)}
        - Evaluation samples: {self.eval_samples}
        - Effective batch size per GPU: {effective_batch_size_per_gpu}
        - Total effective batch size: {total_effective_batch_size}
        - Gradient accumulation steps: {self.gradient_accumulation_steps}
        """)

        # --- Learning Rate and Schedule Recommendation Log (UPDATED) ---
        BASE_BATCH_SIZE_REFERENCE = 256 # A common reference batch size for LLMs
        recommended_base_lr = 3e-4 # A common AdamW LR for smaller/moderate batches

        lr_scaling_factor = total_effective_batch_size / BASE_BATCH_SIZE_REFERENCE
        linearly_scaled_lr = recommended_base_lr * lr_scaling_factor

        # Calculate recommended warmup steps as a percentage of total max steps
        # Use max_steps if you want warmup for the entire run, or cosine_cycle_len for the first cycle
        # Using max_steps for overall recommendation
        recommended_warmup_percentage = 0.05 # 5% is a good starting point
        recommended_warmup_steps = int(self.max_steps * recommended_warmup_percentage)
        
        # Define typical range for cosine annealing cycle length (T_0)
        # This is very task and data dependent. For pre-training, it could be millions.
        # For fine-tuning, often proportional to total training steps.
        recommended_cosine_cycle_len_info = "proportional to total training steps (e.g., 50% to 100% of max_steps for a single long cycle, or shorter cycles for restarts)"

        logging.info(f"Rank {self.rank}: --- Learning Rate and Schedule Recommendations ---")
        logging.info(f"Rank {self.rank}:   Base Batch Size Reference: {BASE_BATCH_SIZE_REFERENCE}, Base LR: {recommended_base_lr:.1e}")
        logging.info(f"Rank {self.rank}:   Your Total Effective Batch Size: {total_effective_batch_size}")
        logging.info(f"Rank {self.rank}:   Linearly scaled peak LR for this batch size: approximately {linearly_scaled_lr:.1e}")
        logging.info(f"Rank {self.rank}:   --> Your configured scheduler_max_lr: {peak_lr:.1e}, scheduler_min_lr: {min_lr:.1e}")
        
        logging.info(f"Rank {self.rank}:   Warmup Steps Recommendation:")
        logging.info(f"Rank {self.rank}:     A typical warmup phase is 5-10% of total training steps.")
        logging.info(f"Rank {self.rank}:     Given max_steps={self.max_steps}, consider around {recommended_warmup_steps} steps.")
        logging.info(f"Rank {self.rank}:     --> Your configured scheduler_warmup_steps: {warmup_total_iters}")

        logging.info(f"Rank {self.rank}:   Cosine Annealing Cycle Length (T_0) Recommendation:")
        logging.info(f"Rank {self.rank}:     Often set to a significant portion of total training steps (e.g., max_steps) or less for restarts.")
        logging.info(f"Rank {self.rank}:     --> Your configured scheduler_max_steps (T_0): {cosine_cycle_len}")
        logging.info(f"Rank {self.rank}: ----------------------------------------------------")
        # --- END Learning Rate and Schedule Recommendation Log ---
        
        # This line was already there, kept it.
        logging.info(f"Rank {self.rank}: Parameters in optimizer:") 

    def set_model_savepath(self, savepath):
        self.savepath = pathlib.Path(savepath)
        self.savepath.mkdir(exist_ok=True, parents=True)
        logging.info(f"Rank {self.rank}: Model save path set to {self.savepath}")
        return self

    def set_trn_iterator(self, iterator):
        self.trn_iterator = iterator
        logging.info(f"Rank {self.rank}: Training iterator set.")
        return self

    def set_validation_dataloader(self, valdl):
        self.valdl = valdl
        logging.info(f"Rank {self.rank}: Validation dataloader set.")
        return self

    def set_mask_percent(self, mask_percent):
        self.mask_percent = mask_percent
        logging.info(f"Rank {self.rank}: Mask percent set to {mask_percent}")
        return self

    def set_metrics_file(self, metrics_path, overwrite=False):
        self.metrics_path = pathlib.Path(metrics_path)
        if self.rank == 0:
            self.metrics_path.parent.mkdir(exist_ok=True, parents=True)
            if overwrite:
                with open(self.metrics_path, 'w') as f:
                    f.write("type\tbatch\tloss\tlr\tauc\tbac\n") # Added AUC/BAC columns
        logging.info(f"Rank {self.rank}: Metrics file path set to {self.metrics_path}")
        return self

    def _train_batch(self, inp, teach, out):

        # Only zero gradients at the beginning of accumulation cycle
        if self.global_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()

        # Move data to device
        inp, teach, out = inp.to(self.rank), teach.to(self.rank), out.to(self.rank)

        should_sync = (self.global_step + 1) % self.gradient_accumulation_steps == 0
        context = torch.enable_grad() if should_sync else self.model.no_sync()
        
        # Forward pass and loss calculation with autocast
        with context:
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = self.model(inp, teach) # [batch_size, seq_len, vocab_size]
                pred = pred.permute(0, 2, 1).contiguous() # [batch_size, vocab_size, seq_len]
                loss = self.lossfn(self.model.module.parameters(), pred, out)
                loss = loss / self.gradient_accumulation_steps
                logging.info(f"Rank {self.rank}: Forward pass complete for step {self.global_step}")

            # Scale gradients and accumulate
            self.scaler.scale(loss).backward()

        # Only update weights at the end of accumulation cycle
        if should_sync:
            logging.info(f"Rank {self.rank}: Backward pass complete for step {self.global_step // self.gradient_accumulation_steps}, scaling gradients.")
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # step once per optimizer update
            self.scheduler.step()

        self.global_step += 1
        # Return the actual loss value for logging (unscaled by accumulation steps)
        return loss.detach().item() * self.gradient_accumulation_steps

    def _eval_all(self, max_eval_batches: Optional[int] = None) -> Tuple[float, float, float]:
        torch.cuda.empty_cache()
        max_eval_batches = self.eval_samples if max_eval_batches is None else max_eval_batches
        max_eval_batches = min(max_eval_batches, len(self.valdl))
    
        self.model.eval()
        total_loss = 0.0
        num_samples = 0

        all_preds = []
        all_targets = []
        value_token_ids = set(self.tokenizer.value_indexes().values())
        value_token_to_01 = {v: k for k, v in self.tokenizer.value_indexes().items()}

        dist.barrier(device_ids=[self.rank])
        for i, (inp, teach, out) in enumerate(self.valdl):
            logging.info(f"Rank {self.rank}: Evaluating batch {i + 1}/{max_eval_batches} (total batches: {len(self.valdl)})")
            if max_eval_batches is not None and i >= max_eval_batches:
                break

            inp, teach, out = inp.to(self.rank, non_blocking=True), teach.to(self.rank, non_blocking=True), out.to(self.rank, non_blocking=True)
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred = self.model(inp, teach)  # [B, T, V]
                    loss = self.eval_loss(self.model.module.parameters(), pred.permute(0, 2, 1).contiguous(), out)

                predval = pred[:, 2::2, :]  # <-- safely picks 2,4,6,...

                value_preds = predval[:, :, list(value_token_ids)]
                pred_probs = F.softmax(value_preds, dim=-1)  # [B, T/2, V]
                outval = out[:, 2::2].contiguous()  # match predval indexing

                out_flat = outval.view(-1)  # [B*T/2]
                pred_probs_flat = pred_probs.view(-1, pred_probs.size(-1))  # [B*T/2, V]

                mask_flat = torch.isin(out_flat, torch.tensor(list(value_token_ids), device=out.device))  # [B*T/2]

                all_preds.append(pred_probs_flat[mask_flat])  # [N, V]
                all_targets.append(out_flat[mask_flat]) 

                total_loss += loss.item() * inp.size(0)
                num_samples += inp.size(0)

        logging.info(f"Rank {self.rank}: Finished evaluating {num_samples} samples across {len(self.valdl)} batches.")
        total_loss_tensor = torch.tensor(total_loss, device=self.rank)
        num_samples_tensor = torch.tensor(num_samples, device=self.rank)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_samples_tensor, op=dist.ReduceOp.SUM)

        logging.info(f"Rank {self.rank}: Total loss after all_reduce: {total_loss_tensor.item()}, Total samples: {num_samples_tensor.item()}")
        mean_loss = total_loss_tensor.item() / num_samples_tensor.item() if num_samples_tensor.item() != 0 else 0.0

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Share local length and compute max
        local_len = torch.tensor([all_preds.size(0)], device=self.rank)
        lengths = [torch.zeros_like(local_len) for _ in range(dist.get_world_size())]
        dist.all_gather(lengths, local_len)
        max_len = max(l.item() for l in lengths)

        # Pad to max length
        pad = max_len - local_len.item()
        if pad > 0:
            all_preds = F.pad(all_preds, (0, 0, 0, pad))
            all_targets = F.pad(all_targets, (0, pad))

        # Gather
        gathered_preds = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
        gathered_targets = [torch.zeros_like(all_targets) for _ in range(dist.get_world_size())]
        logging.info(f"Rank {self.rank}: Gathering predictions and targets across all ranks. {all_preds.shape}")
        dist.all_gather(gathered_preds, all_preds)
        dist.all_gather(gathered_targets, all_targets)

        # Trim padding
        logging.info(f"Rank {self.rank}: Converting targets to binary values.")
        true_total = sum(l.item() for l in lengths)
        all_preds = torch.cat(gathered_preds, dim=0)[:true_total].cpu().numpy()
        all_targets = torch.cat(gathered_targets, dim=0)[:true_total].cpu().numpy()
        all_pred_targets = [(p,value_token_to_01[t]) for p, t in zip(all_preds, all_targets) if t in value_token_to_01]
        all_preds, all_targets = zip(*all_pred_targets) if all_pred_targets else ([], [])
        all_preds = np.stack(all_preds)
        all_targets = np.array(all_targets)

        logging.info(f"Rank {self.rank}: Calculating AUC and BAC.")
        auc = roc_auc_score(all_targets, all_preds[:, 1])
        bac = balanced_accuracy_score(all_targets, all_preds.argmax(axis=1))
        count_0 = (all_targets == 0).sum()
        count_1 = (all_targets == 1).sum()
        max_pred = all_preds.max()
        min_pred = all_preds.min()
        logging.info(f"num samples: {num_samples}, count_0: {count_0}, count_1: {count_1}, max_pred: {max_pred}, min_pred: {min_pred}")
        
        return {
            'loss': mean_loss,
            'auc': auc,
            'bac': bac
        }

    def start(self):
        logging.info(f"Rank {self.rank}: Starting training loop.")
        epoch = 0

        while self.global_step < self.max_steps:
            
            self.trn_iterator.sampler.set_epoch(epoch)
            logging.info(f"Rank {self.rank}: Starting epoch {epoch}. step {self.global_step}.")
            dist.barrier(device_ids=[self.rank])

            self.model.train()  # Ensure model is in training mode

            for i, (inp, teach, out) in enumerate(self.trn_iterator):
                loss = self._train_batch(inp, teach, out)

                # Log training metrics periodically from rank 0
                if self.global_step % self.gradient_accumulation_steps == 0 and self.rank == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    with open(self.metrics_path, 'a') as f:
                        f.write(f"train\t{self.global_step}\t{loss:.4f}\t{current_lr:.6f}\t0.0\t0.0\n")
                    logging.info(f"Epoch: {epoch}, Step: {self.global_step}, Train Loss: {loss:.4f}, LR: {current_lr:.6f}")

                # EVALUATION
                if self.global_step == self.first_eval or (self.global_step + self.first_eval) % self.eval_every == 0:
                    logging.info(f"Rank {self.rank}: Starting evaluation at step {self.global_step}")
                    torch.cuda.synchronize() # Wait for current GPU operations to complete
                    dist.barrier(device_ids=[self.rank])

                    self.model.eval()  # Switch to eval mode
                    with torch.no_grad():  # Prevent gradient computation during eval
                        evals = self._eval_all(max_eval_batches=self.eval_samples)
                        eval_loss, auc, bac = evals['loss'], evals['auc'], evals['bac']
                        logging.info(f"Rank {self.rank}: Evaluation complete. Loss: {eval_loss:.4f}, AUC: {auc:.4f}, BAC: {bac:.4f}")

                    if self.rank == 0 and eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.model.module.save(self.savepath)
                        logging.info(f"Rank {self.rank}: New best eval loss ({self.best_loss:.4f}), saving best model to {self.savepath}")

                    # Also just save periodically in case of crash
                    if self.rank == 0:
                        periodic_save_path = self.savepath.parent / f"step_{self.global_step}"
                        periodic_save_path.mkdir(exist_ok=True, parents=True)
                        self.model.module.save(periodic_save_path)
                        logging.info(f"Rank {self.rank}: Saving periodic model checkpoint to {periodic_save_path}")

                    # Log evaluation metrics from rank 0
                    if self.rank == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        with open(self.metrics_path, 'a') as f:
                            f.write(f"eval\t{self.global_step}\t{eval_loss:.4f}\t{current_lr:.6f}\t{auc:.4f}\t{bac:.4f}\n")
                            logging.info(f"Epoch: {epoch}, Step: {self.global_step}, Train Loss (last cycle): {loss:.4f}, "
                                f"Eval Loss: {eval_loss:.4f}, BAC: {bac:.4f}, AUC: {auc:.4f}, "
                                f"LR: {current_lr:.6f}")

                    # Ensure all ranks are synced before continuing training
                    dist.barrier(device_ids=[self.rank])
                    self.model.train()  # Switch back to training mode
                    logging.info(f"Rank {self.rank}: Model set back to training mode.")
            
            epoch += 1

        logging.info(f"Rank {self.rank}: Training loop finished after {self.global_step} steps.")

class SelfiesPropertyValueTrainer():

    def __init__(self, model, rank, tokenizer, trn_iterator, batch_size, first_eval=100, eval_every=10000, 
    eval_samples=400, scheduler_warmup_steps=10000, scheduler_max_steps=100000, max_steps=100000,
    scheduler_min_lr=1e-6, scheduler_max_lr=3e-4, effective_accum_batch_size=1024, find_unused_parameters=False):

        self.rank = rank
        self.global_step = 0
        self.trn_iterator = trn_iterator
        self.tokenizer = tokenizer

        torch.cuda.set_device(rank)
        logging.info(f"Rank {self.rank}: Setting CUDA device to {rank}")
        self.model = model.to(rank)
        self.lossfn = self.model.build_stratified_lossfn()
        self.eval_loss = self.model.build_stratified_lossfn()
        
        dist.barrier(device_ids=[self.rank]) # Explicitly passing device_ids
        # self.model = torch.compile(self.model)
        self.model = DDP(self.model, device_ids=[rank], find_unused_parameters=find_unused_parameters)
        self.max_steps = max_steps # Total training steps defined here

        # Add gradient accumulation for effectively larger batch sizes
        self.gradient_accumulation_steps = max(1, effective_accum_batch_size // (batch_size * dist.get_world_size()))
        
        # Calculate effective batch size per GPU
        effective_batch_size_per_gpu = batch_size * self.gradient_accumulation_steps
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        total_effective_batch_size = effective_batch_size_per_gpu * world_size # Use effective_batch_size_per_gpu


        # --- LR and Scheduler Setup (UPDATED) ---
        # Define learning rate parameters clearly
        peak_lr = scheduler_max_lr # Your desired maximum LR (e.g., 3e-4 or 1e-2 from user input)
        min_lr = scheduler_min_lr # Your desired minimum LR (e.g., 1e-6 or 1e-5 from user input)
        warmup_total_iters = scheduler_warmup_steps # Number of steps for linear warmup
        # T_0 for cosine annealing, represents the length of the first cosine cycle.
        # It's usually a significant portion of total training steps.
        cosine_cycle_len = scheduler_max_steps 

        # Optimizer is initialized with the 'start' LR for the SequentialLR (which is min_lr for warmup)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=peak_lr, betas=(0.9, 0.99), weight_decay=1e-2)

        # 1. Warm-up scheduler: Linear increase from min_lr to peak_lr
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=min_lr / peak_lr, # Factor to scale optimizer's initial LR (peak_lr) down to min_lr
            end_factor=1.0,                # Factor to bring optimizer's initial LR (peak_lr) back to peak_lr
            total_iters=warmup_total_iters
        )
        
        # 2. Main Cosine Annealing scheduler (starts after warm-up)
        cosine_decay_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=cosine_cycle_len, # This sets the duration of the first cosine cycle
            T_mult=2,             # Each restart cycle is 2x longer
            eta_min=min_lr        # Min learning rate at end of each cycle
        )

        # Combine them: Warm-up first, then Cosine Annealing
        self.scheduler = SequentialLR(
            self.optimizer, 
            schedulers=[warmup_scheduler, cosine_decay_scheduler], 
            milestones=[warmup_total_iters] # Switch from warmup_scheduler to cosine_decay_scheduler after warmup_total_iters
        )
        # --- END LR and Scheduler Setup ---


        self.metrics_path = None
        self.best_loss = np.inf

        # Update GradScaler initialization with device parameter
        self.scaler = GradScaler()

        # Reduce evaluation frequency but evaluate quickly to trigger model saving
        self.first_eval = first_eval
        self.eval_every = eval_every
        self.eval_samples = eval_samples # Number of batches to use for evaluation

    
        logging.info(f"""Trainer initialized with:
        - Model: {self.model.__class__.__name__}
        - Rank: {self.rank}
        - Training samples: {len(self.trn_iterator.dataset)}
        - Evaluation samples: {self.eval_samples}
        - Effective batch size per GPU: {effective_batch_size_per_gpu}
        - Total effective batch size: {total_effective_batch_size}
        - Gradient accumulation steps: {self.gradient_accumulation_steps}
        """)

        # --- Learning Rate and Schedule Recommendation Log (UPDATED) ---
        BASE_BATCH_SIZE_REFERENCE = 256 # A common reference batch size for LLMs
        recommended_base_lr = 3e-4 # A common AdamW LR for smaller/moderate batches

        lr_scaling_factor = total_effective_batch_size / BASE_BATCH_SIZE_REFERENCE
        linearly_scaled_lr = recommended_base_lr * lr_scaling_factor

        # Calculate recommended warmup steps as a percentage of total max steps
        # Use max_steps if you want warmup for the entire run, or cosine_cycle_len for the first cycle
        # Using max_steps for overall recommendation
        recommended_warmup_percentage = 0.05 # 5% is a good starting point
        recommended_warmup_steps = int(self.max_steps * recommended_warmup_percentage)
        
        # Define typical range for cosine annealing cycle length (T_0)
        # This is very task and data dependent. For pre-training, it could be millions.
        # For fine-tuning, often proportional to total training steps.
        recommended_cosine_cycle_len_info = "proportional to total training steps (e.g., 50% to 100% of max_steps for a single long cycle, or shorter cycles for restarts)"

        logging.info(f"Rank {self.rank}: --- Learning Rate and Schedule Recommendations ---")
        logging.info(f"Rank {self.rank}:   Base Batch Size Reference: {BASE_BATCH_SIZE_REFERENCE}, Base LR: {recommended_base_lr:.1e}")
        logging.info(f"Rank {self.rank}:   Your Total Effective Batch Size: {total_effective_batch_size}")
        logging.info(f"Rank {self.rank}:   Linearly scaled peak LR for this batch size: approximately {linearly_scaled_lr:.1e}")
        logging.info(f"Rank {self.rank}:   --> Your configured scheduler_max_lr: {peak_lr:.1e}, scheduler_min_lr: {min_lr:.1e}")
        
        logging.info(f"Rank {self.rank}:   Warmup Steps Recommendation:")
        logging.info(f"Rank {self.rank}:     A typical warmup phase is 5-10% of total training steps.")
        logging.info(f"Rank {self.rank}:     Given max_steps={self.max_steps}, consider around {recommended_warmup_steps} steps.")
        logging.info(f"Rank {self.rank}:     --> Your configured scheduler_warmup_steps: {warmup_total_iters}")

        logging.info(f"Rank {self.rank}:   Cosine Annealing Cycle Length (T_0) Recommendation:")
        logging.info(f"Rank {self.rank}:     Often set to a significant portion of total training steps (e.g., max_steps) or less for restarts.")
        logging.info(f"Rank {self.rank}:     --> Your configured scheduler_max_steps (T_0): {cosine_cycle_len}")
        logging.info(f"Rank {self.rank}: ----------------------------------------------------")
        # --- END Learning Rate and Schedule Recommendation Log ---
        
        # This line was already there, kept it.
        logging.info(f"Rank {self.rank}: Parameters in optimizer:") 

    def set_model_savepath(self, savepath):
        self.savepath = pathlib.Path(savepath)
        self.savepath.mkdir(exist_ok=True, parents=True)
        logging.info(f"Rank {self.rank}: Model save path set to {self.savepath}")
        return self

    def set_trn_iterator(self, iterator):
        self.trn_iterator = iterator
        logging.info(f"Rank {self.rank}: Training iterator set.")
        return self

    def set_validation_dataloader(self, valdl):
        self.valdl = valdl
        logging.info(f"Rank {self.rank}: Validation dataloader set.")
        return self

    def set_mask_percent(self, mask_percent):
        self.mask_percent = mask_percent
        logging.info(f"Rank {self.rank}: Mask percent set to {mask_percent}")
        return self

    def set_metrics_file(self, metrics_path, overwrite=False):
        self.metrics_path = pathlib.Path(metrics_path)
        if self.rank == 0:
            self.metrics_path.parent.mkdir(exist_ok=True, parents=True)
            if overwrite:
                with open(self.metrics_path, 'w') as f:
                    f.write("type\tbatch\tloss\tlr\tauc\tbac\n") # Added AUC/BAC columns
        logging.info(f"Rank {self.rank}: Metrics file path set to {self.metrics_path}")
        return self

    def _train_batch(self, inp, teach, out):

        # Only zero gradients at the beginning of accumulation cycle
        if self.global_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()

        # Move data to device
        inp, teach, out = inp.to(self.rank), teach.to(self.rank), out.to(self.rank)

        should_sync = (self.global_step + 1) % self.gradient_accumulation_steps == 0
        context = torch.enable_grad() if should_sync else self.model.no_sync()
        
        # Forward pass and loss calculation with autocast
        with context:
            with autocast(device_type='cuda', dtype=torch.bfloat16):
                pred = self.model(inp, teach) # [batch_size, seq_len, vocab_size]
                pred = pred.permute(0, 2, 1).contiguous() # [batch_size, vocab_size, seq_len]
                loss = self.lossfn(self.model.module.parameters(), pred, out)
                loss = loss / self.gradient_accumulation_steps
                logging.info(f"Rank {self.rank}: Forward pass complete for step {self.global_step}")

            # Scale gradients and accumulate
            self.scaler.scale(loss).backward()

        # Only update weights at the end of accumulation cycle
        if should_sync:
            logging.info(f"Rank {self.rank}: Backward pass complete for step {self.global_step // self.gradient_accumulation_steps}, scaling gradients.")
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # step once per optimizer update
            self.scheduler.step()

        self.global_step += 1
        # Return the actual loss value for logging (unscaled by accumulation steps)
        return loss.detach().item() * self.gradient_accumulation_steps

    def _eval_all(self, max_eval_batches: Optional[int] = None) -> Tuple[float, float, float]:
        torch.cuda.empty_cache()
        max_eval_batches = self.eval_samples if max_eval_batches is None else max_eval_batches
        max_eval_batches = min(max_eval_batches, len(self.valdl))
    
        self.model.eval()
        total_loss = 0.0
        num_samples = 0

        all_preds = []
        all_targets = []
        value_token_ids = set(self.tokenizer.value_indexes().values())
        value_token_to_01 = {v: k for k, v in self.tokenizer.value_indexes().items()}

        dist.barrier(device_ids=[self.rank])
        for i, (inp, teach, out) in enumerate(self.valdl):
            logging.info(f"Rank {self.rank}: Evaluating batch {i + 1}/{max_eval_batches} (total batches: {len(self.valdl)})")
            if max_eval_batches is not None and i >= max_eval_batches:
                break

            inp, teach, out = inp.to(self.rank, non_blocking=True), teach.to(self.rank, non_blocking=True), out.to(self.rank, non_blocking=True)
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    pred = self.model(inp, teach)  # [B, T, V]
                    loss = self.eval_loss(self.model.module.parameters(), pred.permute(0, 2, 1).contiguous(), out)

                predval = pred[:, 2::2, :]  # <-- safely picks 2,4,6,...

                value_preds = predval[:, :, list(value_token_ids)]
                pred_probs = F.softmax(value_preds, dim=-1)  # [B, T/2, V]
                outval = out[:, 2::2].contiguous()  # match predval indexing

                out_flat = outval.view(-1)  # [B*T/2]
                pred_probs_flat = pred_probs.view(-1, pred_probs.size(-1))  # [B*T/2, V]

                mask_flat = torch.isin(out_flat, torch.tensor(list(value_token_ids), device=out.device))  # [B*T/2]

                all_preds.append(pred_probs_flat[mask_flat])  # [N, V]
                all_targets.append(out_flat[mask_flat]) 

                total_loss += loss.item() * inp.size(0)
                num_samples += inp.size(0)

        logging.info(f"Rank {self.rank}: Finished evaluating {num_samples} samples across {len(self.valdl)} batches.")
        total_loss_tensor = torch.tensor(total_loss, device=self.rank)
        num_samples_tensor = torch.tensor(num_samples, device=self.rank)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_samples_tensor, op=dist.ReduceOp.SUM)

        logging.info(f"Rank {self.rank}: Total loss after all_reduce: {total_loss_tensor.item()}, Total samples: {num_samples_tensor.item()}")
        mean_loss = total_loss_tensor.item() / num_samples_tensor.item() if num_samples_tensor.item() != 0 else 0.0

        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Share local length and compute max
        local_len = torch.tensor([all_preds.size(0)], device=self.rank)
        lengths = [torch.zeros_like(local_len) for _ in range(dist.get_world_size())]
        dist.all_gather(lengths, local_len)
        max_len = max(l.item() for l in lengths)

        # Pad to max length
        pad = max_len - local_len.item()
        if pad > 0:
            all_preds = F.pad(all_preds, (0, 0, 0, pad))
            all_targets = F.pad(all_targets, (0, pad))

        # Gather
        gathered_preds = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
        gathered_targets = [torch.zeros_like(all_targets) for _ in range(dist.get_world_size())]
        logging.info(f"Rank {self.rank}: Gathering predictions and targets across all ranks. {all_preds.shape}")
        dist.all_gather(gathered_preds, all_preds)
        dist.all_gather(gathered_targets, all_targets)

        # Trim padding
        logging.info(f"Rank {self.rank}: Converting targets to binary values.")
        true_total = sum(l.item() for l in lengths)
        all_preds = torch.cat(gathered_preds, dim=0)[:true_total].cpu().numpy()
        all_targets = torch.cat(gathered_targets, dim=0)[:true_total].cpu().numpy()
        all_pred_targets = [(p,value_token_to_01[t]) for p, t in zip(all_preds, all_targets) if t in value_token_to_01]
        all_preds, all_targets = zip(*all_pred_targets) if all_pred_targets else ([], [])
        all_preds = np.stack(all_preds)
        all_targets = np.array(all_targets)

        logging.info(f"Rank {self.rank}: Calculating AUC and BAC.")
        auc = roc_auc_score(all_targets, all_preds[:, 1])
        bac = balanced_accuracy_score(all_targets, all_preds.argmax(axis=1))
        count_0 = (all_targets == 0).sum()
        count_1 = (all_targets == 1).sum()
        max_pred = all_preds.max()
        min_pred = all_preds.min()
        logging.info(f"num samples: {num_samples}, count_0: {count_0}, count_1: {count_1}, max_pred: {max_pred}, min_pred: {min_pred}")
        
        return {
            'loss': mean_loss,
            'auc': auc,
            'bac': bac
        }

    def start(self):
        logging.info(f"Rank {self.rank}: Starting training loop.")
        epoch = 0

        while self.global_step < self.max_steps:
            
            self.trn_iterator.sampler.set_epoch(epoch)
            logging.info(f"Rank {self.rank}: Starting epoch {epoch}. step {self.global_step}.")
            dist.barrier(device_ids=[self.rank])

            self.model.train()  # Ensure model is in training mode

            for i, (inp, teach, out) in enumerate(self.trn_iterator):
                loss = self._train_batch(inp, teach, out)

                # Log training metrics periodically from rank 0
                if self.global_step % self.gradient_accumulation_steps == 0 and self.rank == 0:
                    current_lr = self.optimizer.param_groups[0]['lr']
                    with open(self.metrics_path, 'a') as f:
                        f.write(f"train\t{self.global_step}\t{loss:.4f}\t{current_lr:.6f}\t0.0\t0.0\n")
                    logging.info(f"Epoch: {epoch}, Step: {self.global_step}, Train Loss: {loss:.4f}, LR: {current_lr:.6f}")

                # EVALUATION
                if self.global_step == self.first_eval or (self.global_step + self.first_eval) % self.eval_every == 0:
                    logging.info(f"Rank {self.rank}: Starting evaluation at step {self.global_step}")
                    torch.cuda.synchronize() # Wait for current GPU operations to complete
                    dist.barrier(device_ids=[self.rank])

                    self.model.eval()  # Switch to eval mode
                    with torch.no_grad():  # Prevent gradient computation during eval
                        evals = self._eval_all(max_eval_batches=self.eval_samples)
                        eval_loss, auc, bac = evals['loss'], evals['auc'], evals['bac']
                        logging.info(f"Rank {self.rank}: Evaluation complete. Loss: {eval_loss:.4f}, AUC: {auc:.4f}, BAC: {bac:.4f}")

                    if self.rank == 0 and eval_loss < self.best_loss:
                        self.best_loss = eval_loss
                        self.model.module.save(self.savepath)
                        logging.info(f"Rank {self.rank}: New best eval loss ({self.best_loss:.4f}), saving best model to {self.savepath}")

                    # Also just save periodically in case of crash
                    if self.rank == 0:
                        periodic_save_path = self.savepath.parent / f"step_{self.global_step}"
                        periodic_save_path.mkdir(exist_ok=True, parents=True)
                        self.model.module.save(periodic_save_path)
                        logging.info(f"Rank {self.rank}: Saving periodic model checkpoint to {periodic_save_path}")

                    # Log evaluation metrics from rank 0
                    if self.rank == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        with open(self.metrics_path, 'a') as f:
                            f.write(f"eval\t{self.global_step}\t{eval_loss:.4f}\t{current_lr:.6f}\t{auc:.4f}\t{bac:.4f}\n")
                            logging.info(f"Epoch: {epoch}, Step: {self.global_step}, Train Loss (last cycle): {loss:.4f}, "
                                f"Eval Loss: {eval_loss:.4f}, BAC: {bac:.4f}, AUC: {auc:.4f}, "
                                f"LR: {current_lr:.6f}")

                    # Ensure all ranks are synced before continuing training
                    dist.barrier(device_ids=[self.rank])
                    self.model.train()  # Switch back to training mode
                    logging.info(f"Rank {self.rank}: Model set back to training mode.")
            
            epoch += 1

        logging.info(f"Rank {self.rank}: Training loop finished after {self.global_step} steps.")

