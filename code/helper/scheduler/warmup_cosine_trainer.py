from torch.optim.lr_scheduler import LinearLR, CosineAnnealingWarmRestarts, SequentialLR
import torch


class WarmupCosineScheduler:
    """
    Creates a learning rate scheduler with linear warmup followed by cosine annealing with restarts.
    """
    
    def __init__(self, optimizer, warmup_steps, cosine_cycle_length, min_lr, max_lr, cosine_t_mult=2):
        """
        Args:
            optimizer: PyTorch optimizer
            warmup_steps: Number of steps for linear warmup
            cosine_cycle_length: Length of first cosine cycle (T_0)
            min_lr: Minimum learning rate
            max_lr: Maximum learning rate (optimizer should be initialized with this)
            cosine_t_mult: Multiplier for subsequent cycle lengths
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.cosine_cycle_length = cosine_cycle_length
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.cosine_t_mult = cosine_t_mult
        
        # Create the scheduler
        self._build_scheduler()
    
    def _build_scheduler(self):
        """Build the composite scheduler."""
        # 1. Linear warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer, 
            start_factor=self.min_lr / self.max_lr,  # Start at min_lr
            end_factor=1.0,                          # End at max_lr  
            total_iters=self.warmup_steps
        )
        
        # 2. Cosine annealing with warm restarts
        cosine_scheduler = CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.cosine_cycle_length,
            T_mult=self.cosine_t_mult,
            eta_min=self.min_lr
        )
        
        # 3. Combine them
        self.scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.warmup_steps]
        )
    
    def step(self):
        """Step the scheduler."""
        self.scheduler.step()
    
    def get_last_lr(self):
        """Get the last learning rate."""
        return self.scheduler.get_last_lr()
    
    def state_dict(self):
        """Get scheduler state dict."""
        return self.scheduler.state_dict()
    
    def load_state_dict(self, state_dict):
        """Load scheduler state dict."""
        self.scheduler.load_state_dict(state_dict)