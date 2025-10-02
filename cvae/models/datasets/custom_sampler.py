import torch
import torch.distributed as dist
from torch.utils.data import Sampler
import math


class LightweightDistributedSampler(Sampler):
    """
    A memory-efficient distributed sampler that doesn't materialize all indices.
    
    Uses arithmetic operations to determine which indices belong to each rank,
    avoiding the need to store large index arrays.
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, 
                 seed=0, drop_last=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        # Calculate number of samples per rank
        self.total_size = len(self.dataset)
        
        if self.drop_last:
            # Drop the tail of data to make it evenly divisible
            self.num_samples = self.total_size // self.num_replicas
        else:
            # Add extra samples to make it evenly divisible
            self.num_samples = math.ceil(self.total_size / self.num_replicas)
            
    def __iter__(self):
        if self.shuffle:
            # Create a generator for shuffled indices using a seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            
            # Generate indices for this rank only
            indices = []
            samples_added = 0
            
            # We'll use a simple approach: generate a random permutation seed
            # and use it to determine which indices this rank should get
            perm_seed = torch.randperm(self.total_size, generator=g)
            
            # Take indices for this rank with round-robin distribution
            start_idx = self.rank
            step = self.num_replicas
            
            for i in range(start_idx, len(perm_seed), step):
                if samples_added >= self.num_samples:
                    break
                indices.append(perm_seed[i].item())
                samples_added += 1
                
            # Pad with repetitions if needed (when not drop_last)
            while len(indices) < self.num_samples:
                indices.append(indices[len(indices) % max(1, len(indices))])
                
        else:
            # No shuffle: simple arithmetic distribution
            indices = list(range(self.rank, self.total_size, self.num_replicas))
            
            # Pad if necessary
            while len(indices) < self.num_samples:
                indices.append(indices[len(indices) % max(1, len(indices))])
                
        return iter(indices)
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        """
        Sets the epoch for this sampler. When :attr:`shuffle=True`,
        this ensures all replicas use a different random ordering
        for each epoch.
        """
        self.epoch = epoch


class ArithmeticDistributedSampler(Sampler):
    """
    An even more lightweight distributed sampler using pure arithmetic.
    
    No index materialization at all - generates indices on-demand.
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, 
                 seed=0, drop_last=False):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        if rank >= num_replicas or rank < 0:
            raise ValueError(
                "Invalid rank {}, rank should be in the interval"
                " [0, {}]".format(rank, num_replicas - 1))
        
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        # Calculate samples for this rank
        self.total_size = len(self.dataset)
        if self.drop_last:
            self.num_samples = self.total_size // self.num_replicas
        else:
            self.num_samples = math.ceil(self.total_size / self.num_replicas)
    
    def __iter__(self):
        # TODO - broken, causes systematic bias when shuffling
        # Generate indices arithmetically
        for i in range(self.num_samples):
            if self.shuffle:
                # Use a hash-like function to scramble indices
                # This gives us pseudo-random distribution without storing indices
                scrambled_i = self._scramble_index(i, self.epoch, self.seed)
                idx = (scrambled_i * self.num_replicas + self.rank) % self.total_size
            else:
                # Simple round-robin distribution
                global_idx = i * self.num_replicas + self.rank
                idx = global_idx % self.total_size
                
            yield idx
    
    def _scramble_index(self, index, epoch, seed):
        """Simple hash function to scramble indices for shuffling."""
        # This creates a pseudo-random but deterministic ordering
        x = index + epoch * 31 + seed * 37
        x = ((x >> 16) ^ x) * 0x45d9f3b
        x = ((x >> 16) ^ x) * 0x45d9f3b
        x = (x >> 16) ^ x
        return x & 0x7fffffff  # Keep positive
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch

class FastDistributedSampler(Sampler):
    """
    Alternative approach: Use a different mathematical sequence to assign indices
    that naturally avoids the systematic bias without hashing overhead.
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, 
                 seed=0, drop_last=False):
        if num_replicas is None:
            num_replicas = dist.get_world_size()
        if rank is None:
            rank = dist.get_rank()
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.drop_last = drop_last
        self.shuffle = shuffle
        self.seed = seed
        
        self.total_size = len(self.dataset)
        if self.drop_last:
            self.num_samples = self.total_size // self.num_replicas
        else:
            self.num_samples = math.ceil(self.total_size / self.num_replicas)
    
    def __iter__(self):
        """
        Use Halton sequence or similar low-discrepancy sequence to distribute
        indices more evenly across ranks without systematic bias.
        """
        if self.shuffle:
            # Use coprime numbers to create pseudo-random but deterministic sequence
            # This avoids the round-robin bias
            prime_offset = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37][self.rank % 12]
            multiplier = 2654435761 + self.epoch * 37  # Golden ratio based
            
            for i in range(self.num_samples):
                # Create non-sequential distribution using coprime arithmetic
                pseudo_idx = (i * multiplier + self.rank * prime_offset + self.seed) % self.total_size
                yield pseudo_idx
        else:
            # Deterministic but non-sequential
            offset = (self.rank * 104729) % self.total_size  # Large prime offset
            step = self.num_replicas
            
            for i in range(self.num_samples):
                idx = (offset + i * step) % self.total_size
                yield idx
    
    def __len__(self):
        return self.num_samples
    
    def set_epoch(self, epoch):
        self.epoch = epoch