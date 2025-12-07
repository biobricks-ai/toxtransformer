import sys, os, shutil
import numpy as np
import itertools
import pathlib
import tqdm
import torch
import torch.utils.data
import torch.optim as optim
import cvae.tokenizer
import cvae.utils as utils
import cvae.models.multitask_transformer as mt
import cvae.models.mixture_experts as me
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import logging

logdir = pathlib.Path("cache/train_multitask_transformer_parallel/logs")
logdir.mkdir(exist_ok=True)
cvae.utils.setup_logging(logdir / "log.txt", logging)

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

class Trainer():
    def __init__(self, model, rank, tokenizer, max_epochs=10000):
        self.rank = rank
        self.model = DDP(model.to(rank), device_ids=[rank], find_unused_parameters=True) 
        # self.model = DDP(model.to(rank), device_ids=[rank])
        self.optimizer = optim.AdamW(
            model.parameters(),
            lr=1e-3,  # Moderate initial learning rate
            betas=(0.9, 0.98),  # Transformer defaults
            eps=1e-8,
            weight_decay=0.01  # Add weight decay for regularization
        )
        # self.lossfn = mt.MultitaskTransformer.lossfn(ignore_index=tokenizer.pad_idx)
        self.lossfn = self.model.module.build_lossfn()
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.7,        # Less aggressive reduction (was 0.5)
            patience=10,        # Reduced from 5 to respond faster
            min_lr=1e-6,
            cooldown=2,        # Add cooldown period to stabilize between reductions
            verbose=True       # Print when learning rate changes
        )
        self.max_epochs = max_epochs
        self.metrics_path = None
        self.best_loss = np.inf
        self.tokenizer=tokenizer
    
    def set_model_savepath(self, savepath):
        self.savepath = pathlib.Path(savepath)
        self.savepath.mkdir(exist_ok=True)
        return self

    def set_trn_iterator(self, iterator):
        self.trn_iterator = iterator
        return self

    def set_validation_dataloader(self, valdl):
        self.valdl = valdl
        return self

    def set_mask_percent(self, mask_percent):
        self.mask_percent = mask_percent
        return self

    def set_metrics_file(self, metrics_path, overwrite=False):
        if self.rank == 0:
            self.metrics_path = metrics_path
            if overwrite:
                with open(self.metrics_path, 'w') as f:
                    f.write("type\tbatch\tloss\tlr\n")
        return self

    def _evaluation_loss(self):
        self.model.eval()
        total_loss = 0.0
        num_samples = 0
        
        dist.barrier()
        for i, (inp, teach, out) in enumerate(self.valdl):
            inp, teach, out = inp.to(self.rank), teach.to(self.rank), out.to(self.rank)
            with torch.no_grad():
                pred = self.model(inp, teach)
                loss = self.lossfn(self.model.module.parameters(), pred.permute(0, 2, 1), out)
                total_loss += loss.item() * inp.size(0)
                num_samples += inp.size(0)
        
        # Aggregate the total loss and number of samples across all ranks
        total_loss_tensor = torch.tensor(total_loss, device=self.rank)
        num_samples_tensor = torch.tensor(num_samples, device=self.rank)
        
        dist.all_reduce(total_loss_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(num_samples_tensor, op=dist.ReduceOp.SUM)
        
        if num_samples_tensor.item() != 0:
            mean_loss = total_loss_tensor.item() / num_samples_tensor.item()
        else:
            mean_loss = 0.
        return mean_loss

    def _train_batch(self, inp, teach, out):
        self.optimizer.zero_grad()
        inp, teach, out = inp.to(self.rank), teach.to(self.rank), out.to(self.rank)
        pred = self.model(inp, teach)
        loss = self.lossfn(self.model.module.parameters(), pred.permute(0, 2, 1), out)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        return loss.detach().item()
    
    def _evaluate_balanced_accuracy(self):
        self.model.eval()
        all_preds = []
        all_targets = []
        
        dist.barrier()
        for i, (inp, teach, out) in enumerate(self.valdl):
            inp, teach, out = inp.to(self.rank), teach.to(self.rank), out.to(self.rank)
            with torch.no_grad():
                pred = self.model(inp, teach)
                pred = pred.argmax(dim=2)
                all_preds.append(pred)
                all_targets.append(out)
        
        # Concatenate predictions and targets
        all_preds = torch.cat(all_preds, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Gather predictions and targets from all processes
        gathered_preds = [torch.zeros_like(all_preds) for _ in range(dist.get_world_size())]
        gathered_targets = [torch.zeros_like(all_targets) for _ in range(dist.get_world_size())]
        
        dist.all_gather(gathered_preds, all_preds)
        dist.all_gather(gathered_targets, all_targets)
        
        if self.rank == 0:
            # Combine gathered tensors
            all_preds = torch.cat(gathered_preds, dim=0)
            all_targets = torch.cat(gathered_targets, dim=0)
            
            # Calculate balanced accuracy
            unique_classes = torch.unique(all_targets)
            class_accuracies = []
            
            for cls in unique_classes:
                mask = all_targets == cls
                if mask.sum() > 0:
                    class_acc = (all_preds[mask] == all_targets[mask]).float().mean()
                    class_accuracies.append(class_acc)
            
            balanced_accuracy = torch.stack(class_accuracies).mean().item()
            return balanced_accuracy
        return 0.0


    def start(self):
        total_batches = len(self.trn_iterator)
        self.validation_interval = total_batches // 4  # Evaluate four times per epoch
        for epoch in range(self.max_epochs):
            self.model.train()  # Ensure model is in training mode
            self.trn_iterator.sampler.set_epoch(epoch)
            
            for i, (inp, teach, out) in enumerate(self.trn_iterator):
                loss = self._train_batch(inp, teach, out)
                
                if (i+1) % 2000 == 0:
                    # Ensure all processes are synced before evaluation
                    torch.cuda.synchronize()
                    dist.barrier()
                    
                    self.model.eval()  # Switch to eval mode
                    with torch.no_grad():  # Prevent gradient computation during eval
                        eval_loss = self._evaluation_loss()
                        bac = self._evaluate_balanced_accuracy()
                    
                    self.scheduler.step(eval_loss)
                    
                    if self.rank == 0:
                        if eval_loss < self.best_loss:
                            self.best_loss = eval_loss
                            self.model.module.save(self.savepath)
                            
                        with open(self.metrics_path, 'a') as f:
                            f.write(f"eval\t{i}\t{eval_loss:.4f}\t{self.optimizer.param_groups[0]['lr']}\n")
                            print(f"Epoch: {epoch}, Step: {i}, Train Loss: {loss:.4f}, "
                                f"Eval Loss: {eval_loss:.4f}, LR: {self.optimizer.param_groups[0]['lr']}, "
                                f"BAC: {bac:.4f}")
                    
                    self.model.train()  # Switch back to training mode
                    dist.barrier()  # Ensure all processes are synced before continuing

                elif self.rank == 0:
                    with open(self.metrics_path, 'a') as f:
                        f.write(f"train\t{i}\t{loss:.4f}\t{self.optimizer.param_groups[0]['lr']}\n")


def main(rank, world_size):

    outdir = pathlib.Path("cache/train_multitask_transformer_parallel")
    outdir.mkdir(exist_ok=True)
    
    metricsdir = outdir / "metrics"
    metricsdir.mkdir(exist_ok=True)

    setup(rank, world_size)
    tokenizer = cvae.tokenizer.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

    # model = me.MoE(tokenizer, num_experts=16, hdim=32, dim_feedforward=32, nhead=8, expert_layers=4) # /home/tomlue/git/ai.biobricks/chemprop-transformer/notebook/plots/multitask_transformer_metrics.1.png
    model = me.MoE(tokenizer, num_experts=16, hdim=32*8, dim_feedforward=32*8, nhead=4, balance_loss_weight=.1, expert_layers=6)
    # model = me.MoE.load(outdir / "moe")
    # model = me.MoE.load("brick/moe")
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    
    trnds = mt.SequenceShiftDataset("cache/build_tensordataset/multitask_tensors/trn", tokenizer, nprops=20)
    trndl = torch.utils.data.DataLoader(
        trnds, batch_size=16*4, shuffle=False, num_workers=4, pin_memory=True,
        sampler=torch.utils.data.distributed.DistributedSampler(trnds, num_replicas=world_size, rank=rank)
    )
    
    valds = mt.SequenceShiftDataset("cache/build_tensordataset/multitask_tensors/tst", tokenizer, nprops=20)
    valdl = torch.utils.data.DataLoader(valds, batch_size=16*4, shuffle=False, num_workers=4, pin_memory=True,
        sampler=torch.utils.data.distributed.DistributedSampler(valds, num_replicas=world_size, rank=rank))

    trainer = Trainer(model, rank, tokenizer, max_epochs=10000)\
        .set_trn_iterator(trndl)\
        .set_validation_dataloader(valdl)\
        .set_mask_percent(0.1)\
        .set_model_savepath(outdir / "moe")\
        .set_metrics_file(metricsdir / "multitask_loss.tsv", overwrite=True)
    
    if rank ==0:
        print(f"{len(trndl)} train batches")
        print(f"Process {rank} has {len(valdl)} batches to validate.")
        print(f"{num_params/1e6} million params")
        
    trainer.start()

    cleanup()

    if os.path.exists("brick/moe"):
        shutil.rmtree("brick/moe")
    shutil.copytree(outdir / "moe", "brick/moe")

if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(main, args=(world_size,), nprocs=world_size, join=True)