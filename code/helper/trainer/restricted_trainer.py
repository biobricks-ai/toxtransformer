"""
RestrictedTrainer - A specialized trainer for evaluating specific property tokens.

This module extends the base Trainer class with property-specific evaluation capabilities,
focusing loss calculation and metrics only on the specified property tokens.
"""

import torch
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from helper.trainer.trainer import Trainer


class RestrictedEvalMixin:
    """
    A mixin class that restricts evaluation metrics to specific property tokens.
    Assumes property-value pairs in sequential positions (<p1> <v1> <p2> <v2> etc).
    """
    
    def set_evaluation_properties(self, property_tokens):
        """Set the specific property tokens to evaluate."""
        self.eval_property_tokens = set(property_tokens)
        self.log(f"Rank {self.rank}: Restricted evaluation set to properties {property_tokens}")
        return self
    
    def _eval_all(self, max_eval_batches=None):
        """
        Evaluate the model, restricting loss and metrics to specified property tokens.
        
        Simplifies evaluation by:
        1. Running forward pass on full batch
        2. Filtering to property positions
        3. Masking out properties not in our target set
        4. Computing loss and metrics only on the relevant subset
        """
        if not hasattr(self, 'eval_property_tokens'):
            self.log(f"Rank {self.rank}: No evaluation properties specified.")
            return super()._eval_all(max_eval_batches)

        max_eval_batches = self.eval_samples if max_eval_batches is None else max_eval_batches
        self.model.eval()
        
        # Get value tokens information
        value_token_ids = list(self.tokenizer.value_indexes().values())
        value_token_to_01 = {v: k for k, v in self.tokenizer.value_indexes().items()}
        
        # Collection arrays for metrics
        all_prop_preds = []
        all_prop_targets = []
        total_loss = 0.0
        num_samples = 0
        
        # Property-specific collections
        prop_preds = {p: [] for p in self.eval_property_tokens}
        prop_targets = {p: [] for p in self.eval_property_tokens}
        
        dist.barrier()
        for i, (inp, teach, out) in enumerate(self.valdl):
            if max_eval_batches is not None and i >= max_eval_batches:
                break

            batch_size = inp.size(0)
            inp, teach, out = inp.to(self.rank), teach.to(self.rank), out.to(self.rank)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    # Forward pass
                    pred = self.model(inp, teach)  # [B, T, V]
                    
                    # Extract property positions (0, 2, 4, ...) and their corresponding value positions (1, 3, 5, ...)
                    max_props = (out.size(1) - 1) // 2  # Max number of property-value pairs
                    
                    # Process each property position
                    batch_loss = 0
                    for prop_idx in range(max_props):
                        # Get property and corresponding value positions
                        prop_pos = prop_idx * 2
                        val_pos = prop_pos + 1
                        
                        # Skip if we've reached the end of the sequence
                        if val_pos >= out.size(1):
                            continue
                        
                        # Get property tokens at this position
                        prop_tokens = out[:, prop_pos]  # [B]
                        
                        # Create mask for target properties
                        prop_mask = torch.zeros_like(prop_tokens, dtype=torch.bool)
                        for p in self.eval_property_tokens:
                            prop_mask |= (prop_tokens == p)
                        
                        # Skip if no target properties at this position
                        if not prop_mask.any():
                            continue
                            
                        # Get value tokens and predictions for masked positions
                        val_tokens = out[:, val_pos][prop_mask]  # [filtered_B]
                        val_preds = pred[prop_mask][:, val_pos]  # [filtered_B, V]
                        
                        # Only keep predictions for value tokens (0/1 values)
                        val_preds = val_preds[:, value_token_ids]  # [filtered_B, 2]
                        
                        # Compute loss for this position (cross-entropy with value tokens)
                        val_indices = torch.tensor([value_token_ids.index(v.item()) for v in val_tokens], 
                                                 device=self.rank)
                        pos_loss = F.cross_entropy(val_preds, val_indices)
                        batch_loss += pos_loss * prop_mask.sum().item()
                        
                        # Store predictions and targets for metrics
                        val_probs = F.softmax(val_preds, dim=1)  # [filtered_B, 2]
                        all_prop_preds.append(val_probs)
                        all_prop_targets.append(val_tokens)
                        
                        # Also store by property type
                        for prop_id in self.eval_property_tokens:
                            prop_specific_mask = (prop_tokens == prop_id)
                            if prop_specific_mask.any():
                                prop_val_tokens = out[:, val_pos][prop_specific_mask]
                                prop_val_preds = pred[prop_specific_mask][:, val_pos][:, value_token_ids]
                                prop_val_probs = F.softmax(prop_val_preds, dim=1)
                                
                                prop_preds[prop_id].append(prop_val_probs)
                                prop_targets[prop_id].append(prop_val_tokens)
                    
                    # Normalize batch loss
                    if batch_loss > 0:
                        batch_loss /= batch_size
                        total_loss += batch_loss.item() * batch_size
                        num_samples += batch_size

        # Aggregate loss across all GPUs
        total_loss_tensor = torch.tensor(total_loss, device=self.rank)
        num_samples_tensor = torch.tensor(num_samples, device=self.rank)
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_samples_tensor, op=dist.ReduceOp.SUM)
        
        mean_loss = total_loss_tensor.item() / num_samples_tensor.item() if num_samples_tensor.item() > 0 else 0
        
        # Combine all predictions and targets
        if all_prop_preds:
            all_preds = torch.cat(all_prop_preds, dim=0)
            all_targets = torch.cat(all_prop_targets, dim=0)
            
            # Gather from all GPUs
            gathered_preds = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
            gathered_targets = [torch.zeros_like(all_targets) for _ in range(dist.get_world_size())]
            dist.all_gather(gathered_preds, all_preds)
            dist.all_gather(gathered_targets, all_targets)
            
            all_preds = torch.cat(gathered_preds, dim=0).cpu().numpy()
            all_targets = torch.cat(gathered_targets, dim=0).cpu().numpy()
            
            # Convert targets to binary
            all_targets_binary = np.array([value_token_to_01[t] for t in all_targets])
            
            # Calculate overall metrics
            try:
                overall_auc = roc_auc_score(all_targets_binary, all_preds[:, 1])
                overall_bac = balanced_accuracy_score(all_targets_binary, all_preds.argmax(axis=1))
            except Exception as e:
                self.log(f"Error calculating overall metrics: {e}")
                overall_auc = overall_bac = 0.0
        else:
            overall_auc = overall_bac = 0.0
        
        # Calculate per-property metrics
        property_metrics = {}
        valid_aucs = []
        
        for prop_id in self.eval_property_tokens:
            if not prop_preds[prop_id]:
                property_metrics[prop_id] = {'auc': 0.0, 'bac': 0.0, 'num_samples': 0}
                continue
                
            # Combine predictions and targets for this property
            try:
                prop_all_preds = torch.cat(prop_preds[prop_id], dim=0)
                prop_all_targets = torch.cat(prop_targets[prop_id], dim=0)
                
                # Gather from all GPUs
                gathered_preds = [torch.zeros_like(prop_all_preds) for _ in range(dist.get_world_size())]
                gathered_targets = [torch.zeros_like(prop_all_targets) for _ in range(dist.get_world_size())]
                dist.all_gather(gathered_preds, prop_all_preds)
                dist.all_gather(gathered_targets, prop_all_targets)
                
                prop_all_preds = torch.cat(gathered_preds, dim=0).cpu().numpy()
                prop_all_targets = torch.cat(gathered_targets, dim=0).cpu().numpy()
                
                # Convert to binary
                prop_targets_binary = np.array([value_token_to_01[t] for t in prop_all_targets])
                
                # Calculate metrics
                prop_auc = roc_auc_score(prop_targets_binary, prop_all_preds[:, 1])
                prop_bac = balanced_accuracy_score(prop_targets_binary, prop_all_preds.argmax(axis=1))
                
                property_metrics[prop_id] = {
                    'auc': prop_auc,
                    'bac': prop_bac,
                    'num_samples': len(prop_targets_binary),
                    'count_0': (prop_targets_binary == 0).sum(),
                    'count_1': (prop_targets_binary == 1).sum()
                }
                
                valid_aucs.append(prop_auc)
                
                self.log(f"Property {prop_id}: AUC={prop_auc:.4f}, BAC={prop_bac:.4f}, "
                       f"Samples={len(prop_targets_binary)}")
            except Exception as e:
                self.log(f"Error calculating metrics for property {prop_id}: {e}")
                property_metrics[prop_id] = {'auc': 0.0, 'bac': 0.0, 'error': str(e)}
        
        # Calculate average property AUC
        avg_property_auc = np.mean(valid_aucs) if valid_aucs else 0.0
        
        return {
            'loss': mean_loss,
            'auc': avg_property_auc,  # Use property-specific AUC as the main metric
            'bac': overall_bac,
            'property_metrics': property_metrics
        }
        
    def log_property_metrics(self, metrics, step):
        """Log property-specific metrics to file."""
        if self.rank != 0 or not hasattr(self, 'metrics_path'):
            return
            
        property_metrics_path = self.metrics_path.parent / "property_metrics.tsv"
        if not property_metrics_path.exists():
            with open(property_metrics_path, 'w') as f:
                f.write("step\tproperty\tauc\tbac\tnum_samples\tcount_0\tcount_1\n")
        
        try:
            with open(property_metrics_path, 'a') as f:
                for prop, prop_metrics in metrics.items():
                    f.write(f"{step}\t{prop}\t{prop_metrics.get('auc', 0.0):.4f}\t"
                           f"{prop_metrics.get('bac', 0.0):.4f}\t"
                           f"{prop_metrics.get('num_samples', 0)}\t"
                           f"{prop_metrics.get('count_0', 0)}\t"
                           f"{prop_metrics.get('count_1', 0)}\n")
        except Exception as e:
            self.log(f"Error writing property metrics: {e}")


class RestrictedTrainer(RestrictedEvalMixin, Trainer):
    """
    A trainer that extends the base Trainer with property-specific evaluation.
    Only calculates loss and metrics on specified property tokens.
    """
    
    def __init__(self, 
                 model, 
                 rank, 
                 tokenizer, 
                 trn_iterator, 
                 batch_size, 
                 target_properties=None,
                 scheduler_warmup_steps=10000, 
                 scheduler_max_steps=100000, 
                 max_steps=100000):
        """Initialize the RestrictedTrainer."""
        super().__init__(
            model=model,
            rank=rank,
            tokenizer=tokenizer,
            trn_iterator=trn_iterator,
            batch_size=batch_size,
            scheduler_warmup_steps=scheduler_warmup_steps,
            scheduler_max_steps=scheduler_max_steps,
            max_steps=max_steps
        )
        
        if target_properties:
            self.set_evaluation_properties(target_properties)
            self.log(f"Initialized RestrictedTrainer with target properties: {target_properties}")
    
    def start(self):
        """Start the training process with property-specific evaluation."""
        self.log(f"Rank {self.rank}: Starting restricted training loop.")
        try:
            epoch = 0
            best_auc = 0.0
            
            while self.global_step < self.max_steps:
                self.log(f"Rank {self.rank}: Starting epoch {epoch}")
                self.trn_iterator.sampler.set_epoch(epoch)
                try:
                    dist.barrier()
                except Exception as e:
                    self.log(f"Rank {self.rank}: Barrier warning: {e}")
                
                self.model.train()
                
                for i, (inp, teach, out) in enumerate(self.trn_iterator):
                    if self.global_step >= self.max_steps:
                        break
                        
                    loss = self._train_batch(inp, teach, out)
                    
                    # Log training metrics
                    if self.global_step % self.gradient_accumulation_steps == 0 and self.rank == 0:
                        current_lr = self.optimizer.param_groups[0]['lr']
                        if self.metrics_path:
                            try:
                                with open(self.metrics_path, 'a') as f:
                                    f.write(f"train\t{self.global_step}\t{loss:.4f}\t{current_lr:.6f}\t0.0\t0.0\n")
                            except Exception as e:
                                self.log(f"Error writing to metrics file: {e}")
                    
                    # Run evaluation
                    if self.global_step == self.first_eval or self.global_step % self.eval_every == 0:
                        self.log(f"Rank {self.rank}: Evaluating at step {self.global_step}")
                        torch.cuda.synchronize()
                        
                        try:
                            dist.barrier()
                        except Exception as e:
                            self.log(f"Rank {self.rank}: Barrier warning: {e}")
                        
                        self.model.eval()
                        with torch.no_grad():
                            evals = self._eval_all(max_eval_batches=self.eval_samples)
                            
                            eval_loss = evals['loss']
                            prop_auc = evals['auc']
                            property_metrics = evals.get('property_metrics', {})
                            
                            # Log property metrics
                            if self.rank == 0 and property_metrics:
                                self.log_property_metrics(property_metrics, self.global_step)
                        
                        if self.rank == 0:
                            # Save best model
                            if prop_auc > best_auc:
                                best_auc = prop_auc
                                if hasattr(self.model.module, 'save'):
                                    self.log(f"New best AUC ({best_auc:.4f}), saving model")
                                    self.model.module.save(self.savepath)
                            
                            # Periodic checkpoint
                            if self.savepath:
                                periodic_path = self.savepath.parent / f"step_{self.global_step}"
                                if not periodic_path.exists():
                                    periodic_path.mkdir(exist_ok=True, parents=True)
                                if hasattr(self.model.module, 'save'):
                                    self.model.module.save(periodic_path)
                            
                            # Log metrics
                            if self.metrics_path:
                                current_lr = self.optimizer.param_groups[0]['lr']
                                try:
                                    with open(self.metrics_path, 'a') as f:
                                        f.write(f"eval\t{self.global_step}\t{eval_loss:.4f}\t"
                                              f"{current_lr:.6f}\t{prop_auc:.4f}\t0.0\n")
                                except Exception as e:
                                    self.log(f"Error writing metrics: {e}")
                                
                                # Log a summary
                                property_summary = ", ".join(
                                    f"{prop}({metrics['auc']:.3f})" 
                                    for prop, metrics in property_metrics.items()
                                    if metrics.get('num_samples', 0) > 0
                                )
                                
                                self.log(f"Step {self.global_step}, Loss: {eval_loss:.4f}, "
                                       f"Avg AUC: {prop_auc:.4f}, Props: {property_summary}")
                        
                        self.model.train()
                
                epoch += 1
        
        except Exception as e:
            self.log(f"Rank {self.rank}: Error in training loop: {e}")
            import traceback
            self.log(f"Rank {self.rank}: Traceback:\n{traceback.format_exc()}")
        
        self.log(f"Rank {self.rank}: Training finished after {self.global_step} steps.")


def create_restricted_trainer(
    model, rank, tokenizer, train_iterator, batch_size, target_properties,
    scheduler_warmup_steps=10000, scheduler_max_steps=100000, max_steps=100000,
    savepath=None, metrics_path=None, validation_dataloader=None
):
    """Factory function to create and configure a RestrictedTrainer instance."""
    trainer = RestrictedTrainer(
        model=model, rank=rank, tokenizer=tokenizer, trn_iterator=train_iterator,
        batch_size=batch_size, target_properties=target_properties,
        scheduler_warmup_steps=scheduler_warmup_steps,
        scheduler_max_steps=scheduler_max_steps, max_steps=max_steps
    )
    
    if savepath:
        trainer.set_model_savepath(savepath)
    
    if metrics_path:
        trainer.set_metrics_file(metrics_path)
    
    if validation_dataloader:
        trainer.set_validation_dataloader(validation_dataloader)
    
    return trainer