from sklearn.metrics import roc_auc_score, balanced_accuracy_score
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import ReduceLROnPlateau
from lion_pytorch import Lion
import logging
import pathlib
import numpy as np
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
import traceback # Import traceback to log full error info
import datetime
from cvae.models.multitask_transformer import linear_warmup_and_decay_scheduler
from typing import Optional, Tuple

class Trainer():

    def __init__(self, model, rank, tokenizer, trn_iterator, batch_size, first_eval=100, eval_every=10000, 
    eval_samples=400, scheduler_warmup_steps=10000, scheduler_max_steps=100000, max_steps=100000,
    scheduler_min_lr=1e-6, scheduler_max_lr=3e-4, effective_accum_batch_size=1024):
        self.rank = rank
        self.global_step = 0
        self.trn_iterator = trn_iterator
        self.tokenizer = tokenizer
        self.log(f"Rank {rank}: Initializing Trainer.")

        # init model
        self.log(f"Rank {rank}: Moving model to device {rank}")
        self.model = model.to(rank)
        self.log(f"Rank {rank}: Model initialized on device {rank}")

        # Build loss functions *before* DDP and compile
        # Assuming build_stratified_lossfn exists and works on the base model
        self.log(f"Rank {rank}: Attempting to build loss functions.")
        

        self.lossfn = self.model.build_stratified_lossfn()
        self.eval_loss = self.model.build_stratified_lossfn()
        self.log(f"Rank {rank}: Stratified loss functions built successfully.")
        self.log(f"Rank {rank}: Finished attempt to build loss functions.")

        # Ensure all ranks are here before initializing DDP
        self.log(f"Rank {rank}: Entering barrier before DDP initialization.")
        dist.barrier()
        self.log(f"Rank {rank}: Barrier passed, proceeding with DDP initialization.")
        self.log(f"Rank {rank}: Wrapping model with DDP.")
        self.model = torch.compile(self.model)
        self.model = DDP(self.model, device_ids=[rank])
        self.log(f"Rank {rank}: Model DDP wrapped.")

        self.log(f"Rank {rank}: Compiling model.")
        
        self.log(f"Rank {rank}: Model compiled successfully.")

        self.max_steps = max_steps

        # Add gradient accumulation for effectively larger batch sizes
        # self.gradient_accumulation_steps = 10
        self.gradient_accumulation_steps = max(1, effective_accum_batch_size // (batch_size * dist.get_world_size()))
        
        # Calculate effective batch size per GPU
        effective_batch_size_per_gpu = batch_size * self.gradient_accumulation_steps
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        total_effective_batch_size = effective_batch_size_per_gpu * world_size

        # Scale learning rate based on total effective batch size (common practice)
        # Using 128 as a reference batch size for LR scaling
        self.log(f"Rank {rank}: Effective batch size per GPU: {effective_batch_size_per_gpu}")
        self.log(f"Rank {rank}: Total effective batch size: {total_effective_batch_size}")

        self.log(f"Rank {rank}: Initializing optimizer.")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), # Use self.model (DDP wrapped, potentially compiled)
            lr=1,
            betas=(0.9, 0.99),
            weight_decay=1e-2
        )
        self.log(f"Rank {rank}: Optimizer initialized.")

        self.log(f"Rank {rank}: Initializing scheduler.")
        max_lr = scheduler_max_lr
        min_lr = scheduler_min_lr
        self.scheduler = linear_warmup_and_decay_scheduler(self.optimizer, max_lr=max_lr, min_lr=min_lr, warmup_steps=scheduler_warmup_steps, total_steps=scheduler_max_steps)
        self.scheduler.step()
        self.log(f"Rank {rank}: Scheduler initialized. max_lr is {max_lr}, min_lr is {min_lr}, lr is {self.optimizer.param_groups[0]['lr']}, warmup_steps={scheduler_warmup_steps}, total_steps={scheduler_max_steps}, gradient_accumulation_steps={self.gradient_accumulation_steps}, effective_batch_size={effective_batch_size_per_gpu * world_size}")

        self.metrics_path = None
        self.best_loss = np.inf

        # Update GradScaler initialization with device parameter
        self.log(f"Rank {rank}: Initializing GradScaler.")
        self.scaler = GradScaler()
        self.log(f"Rank {rank}: GradScaler initialized.")

        # Reduce evaluation frequency but evaluate quickly to trigger model saving
        self.first_eval = first_eval
        self.eval_every = eval_every
        self.eval_samples = eval_samples # Number of batches to use for evaluation

        # Ensure loss functions were built
        if self.lossfn is None or self.eval_loss is None:
             self.log(f"Rank {rank}: Error: Loss functions were not successfully built during initialization. Training may fail.")
        else:
             self.log(f"Rank {rank}: Loss functions successfully initialized.")


    def log(self, msg):
        # Only log from rank 0
        if self.rank == 0:
            logging.info(msg)

    def set_model_savepath(self, savepath):
        self.savepath = pathlib.Path(savepath)
        if self.rank == 0:
            self.savepath.mkdir(exist_ok=True, parents=True)
        self.log(f"Rank {self.rank}: Model save path set to {self.savepath}")
        return self

    def set_trn_iterator(self, iterator):
        self.trn_iterator = iterator
        self.log(f"Rank {self.rank}: Training iterator set.")
        return self

    def set_validation_dataloader(self, valdl):
        self.valdl = valdl
        self.log(f"Rank {self.rank}: Validation dataloader set.")
        return self

    def set_mask_percent(self, mask_percent):
        self.mask_percent = mask_percent
        self.log(f"Rank {self.rank}: Mask percent set to {mask_percent}")
        return self

    def set_metrics_file(self, metrics_path, overwrite=False):
        self.metrics_path = pathlib.Path(metrics_path)
        if self.rank == 0:
            self.metrics_path.parent.mkdir(exist_ok=True, parents=True)
            if overwrite:
                with open(self.metrics_path, 'w') as f:
                    f.write("type\tbatch\tloss\tlr\tauc\tbac\n") # Added AUC/BAC columns
        self.log(f"Rank {self.rank}: Metrics file path set to {self.metrics_path}")
        return self

    def _train_batch(self, inp, teach, out):
        
        # step once per batch
        self.scheduler.step()

        # Only zero gradients at the beginning of accumulation cycle
        if self.global_step % self.gradient_accumulation_steps == 0:
            self.optimizer.zero_grad()

        # Move data to device
        inp, teach, out = inp.to(self.rank), teach.to(self.rank), out.to(self.rank)

        # Forward pass and loss calculation with autocast
        # self.log(f"Rank {self.rank}: Starting forward pass for step {self.global_step}")
        with autocast(device_type='cuda', dtype=torch.float16):
            pred = self.model(inp, teach) # [batch_size, seq_len, vocab_size]
            pred = pred.permute(0, 2, 1).contiguous() # [batch_size, vocab_size, seq_len]
            loss = self.lossfn(self.model.module.parameters(), pred, out)
            loss = loss / self.gradient_accumulation_steps


        # Scale gradients and accumulate
        self.scaler.scale(loss).backward()

        # Only update weights at the end of accumulation cycle
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.scaler.step(self.optimizer)
            self.scaler.update()

        self.global_step += 1
        # Return the actual loss value for logging (unscaled by accumulation steps)
        return loss.detach().item() * self.gradient_accumulation_steps

    def _eval_all(self, max_eval_batches: Optional[int] = None) -> Tuple[float, float, float]:
        max_eval_batches = self.eval_samples if max_eval_batches is None else max_eval_batches

        self.model.eval()
        total_loss = 0.0
        num_samples = 0

        all_preds = []
        all_targets = []
        value_token_ids = set(self.tokenizer.value_indexes().values())
        value_token_to_01 = {v: k for k, v in self.tokenizer.value_indexes().items()}
        
        dist.barrier()
        for i, (inp, teach, out) in enumerate(self.valdl):
            if max_eval_batches is not None and i >= max_eval_batches:
                break

            inp, teach, out = inp.to(self.rank, non_blocking=True), teach.to(self.rank, non_blocking=True), out.to(self.rank, non_blocking=True)
            with torch.no_grad():
                with autocast(device_type='cuda', dtype=torch.float16):
                    pred = self.model(inp, teach) # [B, T, V]
                    loss = self.eval_loss(self.model.module.parameters(), pred.permute(0,2,1).contiguous(), out)

                value_preds = pred[:, :, list(value_token_ids)]
                pred_probs = F.softmax(value_preds, dim=-1)  # [B, T, V]
                out_flat = out.view(-1)  # [B*T]
                pred_probs_flat = pred_probs.view(-1, pred_probs.size(-1))  # [B*T, V]
                mask_flat = torch.isin(out_flat, torch.tensor(list(value_token_ids), device=out.device))  # [B*T]

                all_preds.append(pred_probs_flat[mask_flat])  # [N, V]
                all_targets.append(out_flat[mask_flat])        # [N]

                total_loss += loss.item() * inp.size(0)
                num_samples += inp.size(0)

        total_loss_tensor = torch.tensor(total_loss, device=self.rank)
        num_samples_tensor = torch.tensor(num_samples, device=self.rank)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_samples_tensor, op=dist.ReduceOp.SUM)

        mean_loss = total_loss_tensor.item() / num_samples_tensor.item() if num_samples_tensor.item() != 0 else 0.0

        if len(all_preds) > 0:
            all_preds = torch.cat(all_preds, dim=0)
            all_targets = torch.cat(all_targets, dim=0)

            max_tokens = 10000
            if all_preds.size(0) > max_tokens:
                self.log(f"Truncating predictions from {all_preds.size(0)} to {max_tokens}")
                all_preds = all_preds[:max_tokens]
                all_targets = all_targets[:max_tokens]

            gathered_preds = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
            gathered_targets = [torch.zeros_like(all_targets) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_preds, all_preds)
            dist.all_gather(gathered_targets, all_targets)

            all_preds = torch.cat(gathered_preds, dim=0).cpu().numpy()
            all_targets = torch.cat(gathered_targets, dim=0).cpu().numpy()

            # change all_targets to be 0 or 1, currently it takes on values in value_token_ids
            all_targets = np.array([value_token_to_01[x] for x in all_targets])

            auc = roc_auc_score(all_targets, all_preds[:, 1])
            bac = balanced_accuracy_score(all_targets, all_preds.argmax(axis=1))
            count_0 = (all_targets == 0).sum()
            count_1 = (all_targets == 1).sum()
            max_pred = all_preds.max()
            min_pred = all_preds.min()
            self.log(f"num samples: {num_samples}, count_0: {count_0}, count_1: {count_1}, max_pred: {max_pred}, min_pred: {min_pred}")
            return {
                'loss': mean_loss,
                'auc': auc,
                'bac': bac
            }

        return {'loss': mean_loss, 'auc': 0.0, 'bac': 0.0}

    def start(self):
        self.log(f"Rank {self.rank}: Starting training loop.")
        # self.populate_property_loss_tracker()
        try:
            epoch = 0
            while self.global_step < self.max_steps:
                self.log(f"Rank {self.rank}: Starting epoch {epoch}")
                self.trn_iterator.sampler.set_epoch(epoch)
                self.log(f"Rank {self.rank}: Sampler epoch set to {epoch}")
                
                # Use timeout for barrier to prevent indefinite hanging
                try:
                    dist.barrier()  # 5-minute timeout
                    self.log(f"Rank {self.rank}: Barrier passed for epoch {epoch}")
                except Exception as e:
                    self.log(f"Rank {self.rank}: Warning: Barrier timeout at epoch start: {e}")
                    # Continue anyway

                self.model.train()  # Ensure model is in training mode
                self.log(f"Rank {self.rank}: Model set to training mode.")

                # Add log before starting batch iteration
                self.log(f"Rank {self.rank}: Starting batch iteration for epoch {epoch}")
                try:
                    for i, (inp, teach, out) in enumerate(self.trn_iterator):
                        if self.global_step >= self.max_steps:
                            break
                            
                        loss = self._train_batch(inp, teach, out)

                        # Log training metrics periodically from rank 0
                        if self.global_step % self.gradient_accumulation_steps == 0 and self.rank == 0:
                            current_lr = self.optimizer.param_groups[0]['lr']
                            if self.metrics_path:
                                try:
                                    with open(self.metrics_path, 'a') as f:
                                        # Placeholder 0s for AUC/BAC in train logs
                                        f.write(f"train\t{self.global_step}\t{loss:.4f}\t{current_lr:.6f}\t0.0\t0.0\n")
                                except Exception as e:
                                    self.log(f"Rank {self.rank}: Error writing to metrics file: {e}")
                            logging.info(f"Epoch: {epoch}, Step: {self.global_step}, Train Loss: {loss:.4f}, LR: {current_lr:.6f}")

                        # Evaluate less frequently to speed up training
                        if self.global_step == self.first_eval or self.global_step % self.eval_every == 0:
                            self.log(f"Rank {self.rank}: Starting evaluation at step {self.global_step}")
                            # Ensure all processes are synced before evaluation
                            torch.cuda.synchronize() # Wait for current GPU operations to complete
                            
                            # Use timeout for barrier to prevent indefinite hanging
                            try:
                                dist.barrier()
                                self.log(f"Rank {self.rank}: Barrier passed before evaluation")
                            except Exception as e:
                                self.log(f"Rank {self.rank}: Warning: Barrier timeout before evaluation: {e}")
                                # Continue anyway

                            self.model.eval()  # Switch to eval mode
                            self.log(f"Rank {self.rank}: Model set to evaluation mode.")
                            with torch.no_grad():  # Prevent gradient computation during eval
                                evals = self._eval_all(max_eval_batches=self.eval_samples)
                                eval_loss, auc, bac = evals['loss'], evals['auc'], evals['bac']
                                self.log(f"Rank {self.rank}: Evaluation complete. Loss: {eval_loss:.4f}, AUC: {auc:.4f}, BAC: {bac:.4f}")

                                # Scheduler step based on evaluation loss (only on rank 0 typically)
                                if self.rank == 0:
                                    # self.scheduler.step(eval_loss)
                                    self.log(f"Rank {self.rank}: Scheduler stepped with eval loss {eval_loss:.4f}. New LR: {self.optimizer.param_groups[0]['lr']:.6f}")

                            if self.rank == 0:
                                # Save best model based on evaluation loss
                                if eval_loss < self.best_loss:
                                    self.best_loss = eval_loss
                                    if hasattr(self.model.module, 'save'):
                                        self.log(f"Rank {self.rank}: New best eval loss ({self.best_loss:.4f}), saving best model to {self.savepath}")
                                        self.model.module.save(self.savepath)
                                    else:
                                        self.log(f"Rank {self.rank}: Warning: Model module does not have a 'save' method. Skipping best model save.")

                                # Also just save periodically in case of crash
                                if self.savepath: # Check if savepath was set
                                    periodic_save_path = self.savepath.parent / f"step_{self.global_step}"
                                    if not periodic_save_path.exists():
                                        periodic_save_path.mkdir(exist_ok=True, parents=True)
                                    if hasattr(self.model.module, 'save'):
                                         self.log(f"Rank {self.rank}: Saving periodic model checkpoint to {periodic_save_path}")
                                         self.model.module.save(periodic_save_path)
                                    else:
                                        self.log(f"Rank {self.rank}: Warning: Model module does not have a 'save' method. Skipping periodic model save.")

                                # Log evaluation metrics from rank 0
                                if self.metrics_path:
                                    current_lr = self.optimizer.param_groups[0]['lr']
                                    try:
                                        with open(self.metrics_path, 'a') as f:
                                            f.write(f"eval\t{self.global_step}\t{eval_loss:.4f}\t{current_lr:.6f}\t{auc:.4f}\t{bac:.4f}\n")
                                    except Exception as e:
                                        self.log(f"Rank {self.rank}: Error writing to metrics file: {e}")
                                    logging.info(f"Epoch: {epoch}, Step: {self.global_step}, Train Loss (last cycle): {loss:.4f}, "
                                        f"Eval Loss: {eval_loss:.4f}, BAC: {bac:.4f}, AUC: {auc:.4f}, "
                                        f"LR: {current_lr:.6f}")

                            # Ensure all ranks are synced before continuing training
                            try:
                                dist.barrier()
                                self.log(f"Rank {self.rank}: Evaluation barrier passed, resuming training.")
                            except Exception as e:
                                self.log(f"Rank {self.rank}: Warning: Barrier timeout after evaluation: {e}")
                                # Continue anyway

                            self.model.train()  # Switch back to training mode
                            self.log(f"Rank {self.rank}: Model set back to training mode.")

                except Exception as e:
                    self.log(f"Rank {self.rank}: Error during training loop at step {self.global_step}: {e}")
                    import traceback
                    self.log(f"Rank {self.rank}: Traceback:\n{traceback.format_exc()}")
                    # Don't exit, try to continue with next epoch
                
                epoch += 1

        except Exception as e:
            self.log(f"Rank {self.rank}: Fatal error in training loop: {e}")
            import traceback
            self.log(f"Rank {self.rank}: Traceback:\n{traceback.format_exc()}")
            # Attempt to clean up
            try:
                dist.barrier()
            except:
                pass
        
        self.log(f"Rank {self.rank}: Training loop finished after {self.global_step} steps.")

