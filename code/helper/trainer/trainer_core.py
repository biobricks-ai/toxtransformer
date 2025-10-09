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
from .evaluator import DefaultEvaluator, Evaluator

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
                 find_unused_parameters: bool = False,
                 evaluator: Optional[Evaluator] = None,
                 save_every: int = 10_000):
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
        self.save_every = save_every

        world = dist.get_world_size() if dist.is_initialized() else 1
        self.gradient_accumulation_steps = max(1, effective_accum_batch_size // (batch_size * world))

        self.scheduler = scheduler
        self.optimizer = scheduler.optimizer

        # Set up evaluator
        self.evaluator = evaluator if evaluator is not None else DefaultEvaluator(rank=rank, tokenizer=tokenizer)

        self.metrics_path = None
        self.best_loss = np.inf
        self.best_auc = 0.0
        self.valdl = None
        self.savepath = None

        # For tracking moving averages
        self.loss_history = []
        self.loss_window = 100  # Track last 100 steps for moving average

        logging.info(
            f"TrainerCore: rank={rank}, grad_accum={self.gradient_accumulation_steps}, "
            f"world={world}, max_steps={max_steps}, save_every={save_every}, evaluator={type(self.evaluator).__name__}"
        )

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

    def set_evaluator(self, evaluator: Evaluator):
        """Set a custom evaluator for this trainer."""
        self.evaluator = evaluator
        if self.rank == 0:
            logging.info(f"Updated evaluator to: {type(evaluator).__name__}")
        return self

    def set_metrics_file(self, metrics_path: str, overwrite: bool = False):
        self.metrics_path = pathlib.Path(metrics_path)
        if self.rank == 0:
            self.metrics_path.parent.mkdir(exist_ok=True, parents=True)
            if overwrite:
                with open(self.metrics_path, 'w') as f:
                    f.write("type\tbatch\tloss\tlr\tauc\tbac\n")
        return self

    def _save_model(self, checkpoint_name: str):
        """Save the model checkpoint."""
        if self.savepath is None:
            logging.warning("No save path set, skipping model save")
            return
            
        checkpoint_path = self.savepath / f"{checkpoint_name}"
        checkpoint_path.mkdir(exist_ok=True, parents=True)
        # Get the actual model (unwrap DDP if necessary)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        
        # Check if the model has a save method
        if not hasattr(model_to_save, 'save'):
            raise AttributeError(f"Model {type(model_to_save).__name__} does not have a 'save' method")
        
        model_to_save.save(checkpoint_path)

    @torch.no_grad()
    def _eval_all(self, max_eval_batches: Optional[int] = None) -> Dict[str, float]:
        """
        Evaluate the model using the configured evaluator.
        
        Args:
            max_eval_batches: Maximum number of batches to evaluate
            
        Returns:
            Dictionary containing evaluation metrics
        """
        assert self.valdl is not None, "Validation dataloader not set"
        
        max_eval_batches = self.eval_samples if max_eval_batches is None else max_eval_batches
        
        return self.evaluator.evaluate(
            model=self.model,
            valdl=self.valdl,
            max_eval_batches=max_eval_batches
        )

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

                if self.global_step >= self.max_steps:
                    break
                
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
                        logging.info(f"Evaluation results: {evals}")
                        is_best_loss = evals['loss'] < self.best_loss
                        is_best_auc = evals['auc'] > self.best_auc
                        
                        if is_best_loss:
                            self.best_loss = evals['loss']
                        if is_best_auc:
                            self.best_auc = evals['auc']
                        
                        if is_best_loss and self.rank == 0:
                            logging.info(f"New best loss: {self.best_loss:.4f} at step {self.global_step}, saving model...")
                            self._save_model("best_loss")

                        improvement_str = " 🎯 BEST LOSS!" if is_best_loss else ""
                        improvement_str += " 🎯 BEST AUC!" if is_best_auc else ""

                        logging.info(
                            f"✅ EVAL COMPLETE [Step {self.global_step}] - "
                            f"Loss: {evals['loss']:.4f} (best: {self.best_loss:.4f}), "
                            f"AUC: {evals['auc']:.4f} (best: {self.best_auc:.4f}), "
                            f"{improvement_str}"
                            # f"BAC: {evals['bac']:.4f}, LR: {current_lr:.2e}"
                        )
                        
                        if self.metrics_path:
                            with open(self.metrics_path, 'a') as f:
                                # Write basic metrics
                                f.write(f"eval\t{self.global_step}\t{evals['loss']:.4f}\t{current_lr:.6f}\t{evals['auc']:.4f}\n")

                    dist.barrier(device_ids=[self.rank])
                    self.model.train()

                # Periodic saving
                if self.save_every > 0 and self.global_step % self.save_every == 0 and self.rank == 0:
                    self._save_model(f"checkpoint_step_{self.global_step}")
                
                self.global_step += 1

            epoch += 1
            logging.info(f"📅 Completed epoch {epoch}")

        logging.info(
            f"🏁 Training finished at step {self.global_step}! "
                f"Best loss: {self.best_loss:.4f}, Best AUC: {self.best_auc:.4f}"
            )
        
        # Save final checkpoint
        if self.rank == 0:
            self._save_model("final_checkpoint")
