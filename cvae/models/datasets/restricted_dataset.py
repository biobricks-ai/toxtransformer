from torch.utils.data import Dataset
import torch
import pathlib, bisect, random, tqdm
import torch.nn.functional as F
from collections import defaultdict
import logging
import threading
import torch.distributed as dist
import logging

# USAGE EXAMPLE
# Initialize with specific property tokens and positions
# tokenizer = YourTokenizer()
# target_properties = [102, 155, 203]  # Property tokens to guarantee
# target_positions = [0, 1, 2]         # Possible positions to place them
#
# # Optional: Define sampling weights for different properties
# weights = {
#     102: 0.5,  # Sample this property 50% of the time
#     155: 0.3,  # Sample this property 30% of the time
#     203: 0.2   # Sample this property 20% of the time
# }
#
# # For multi-GPU training
# dataset = PropertyGuaranteeDataset(
#     path="data/compounds/",
#     tokenizer=tokenizer,
#     target_props=target_properties,
#     target_positions=target_positions, 
#     sampling_weights=weights,
#     distributed=True,
#     rank=dist.get_rank(),
#     world_size=dist.get_world_size()
# )
#
# # Synchronize sample counts across GPUs periodically
# if batch_idx % 100 == 0:
#     dataset.sync_sample_counts()

class SharedSampleTracker:
    """
    Thread-safe tracker for sample frequency across GPU ranks.
    Uses a shared tensor in shared memory for efficient cross-rank communication.
    
    Args:
        size (int): Number of samples to track
        rank (int): Current process rank
        world_size (int): Total number of processes
    """
    def __init__(self, size, rank=0, world_size=1):
        self.size = size
        self.rank = rank
        self.world_size = world_size
        
        # Create shared tensor for tracking sample counts
        self.tensor = torch.zeros(size, dtype=torch.int32)
        if world_size > 1:
            # Move tensor to shared memory for multi-GPU
            self.tensor = self.tensor.share_memory_()
        
        # Thread safety
        self.lock = threading.Lock()
        
    def increment(self, idx):
        """Thread-safe increment of sample count"""
        with self.lock:
            self.tensor[idx] += 1
    
    def get_count(self, idx):
        """Get current count for an index"""
        return self.tensor[idx].item()
    
    def get_counts(self, indices):
        """Get counts for multiple indices"""
        return [self.tensor[idx].item() for idx in indices]
    
    def sync_across_ranks(self):
        """Synchronize counts across all GPU ranks"""
        if self.world_size > 1:
            # Sum counts across all ranks
            dist.all_reduce(self.tensor, op=dist.ReduceOp.SUM)


class PropertyGuaranteeDataset(Dataset):
    """
    A PyTorch Dataset that guarantees specific property tokens appear in samples.
    
    Key Features:
    - Indexes data by property token (not position)
    - Places selected properties at requested positions during sampling
    - Tracks sample frequency across GPU ranks using shared memory
    - Supports weighted sampling across properties
    - Shuffles other properties
    
    Args:
        path (str or Path): Directory with .pt files containing 'selfies' and 'assay_vals'.
        tokenizer: Tokenizer with PAD_IDX, SEP_IDX, END_IDX.
        target_props (list): List of property tokens to target.
        target_positions (list): List of positions where property tokens should be placed.
        sampling_weights (dict, optional): Dict mapping property tokens to sampling weights.
        distributed (bool): Whether to use distributed training mode.
        rank (int): Current process rank for distributed training.
        world_size (int): Total number of processes for distributed training.
    """

    def __init__(self, path, tokenizer, nprops, target_props, target_positions, 
                 sampling_weights=None, distributed=False, rank=0, world_size=1):
        self.path = str(path)
        self.pad_idx, self.sep_idx, self.end_idx = tokenizer.PAD_IDX, tokenizer.SEP_IDX, tokenizer.END_IDX
        self.target_props = set(target_props)  # Set of property tokens to target
        self.target_positions = target_positions  # List of positions where properties should be placed
        self.tokenizer = tokenizer
        self.nprops = nprops

        # Initialize index structures - mapping property tokens to dataset indices
        self.property_index = defaultdict(list)  # Maps prop_token to list of dataset indices
        self.data = []
        self.cumulative_lengths = [0]
        
        # Set up distributed parameters
        self.distributed = distributed
        self.rank = rank
        self.world_size = world_size
        
        # Set up sampling weights (default to uniform if not provided)
        if sampling_weights is None:
            self.sampling_weights = {prop: 1.0 for prop in target_props}
        else:
            self.sampling_weights = sampling_weights
            # Ensure all target props have weights
            for prop in target_props:
                if prop not in self.sampling_weights:
                    self.sampling_weights[prop] = 1.0
            
        # Normalize weights
        total_weight = sum(self.sampling_weights.values())
        self.sampling_weights = {k: v/total_weight for k, v in self.sampling_weights.items()}
        
        # Load and index data
        self._load_and_index_data(path)
        
        # Create sample tracker after knowing dataset size
        self.sample_tracker = SharedSampleTracker(
            size=len(self), 
            rank=rank if distributed else 0,
            world_size=world_size if distributed else 1
        )
        
        # Verify we have examples for each target property
        self._validate_targets()
        
        logging.info(f"PropertyGuaranteeDataset initialized with {len(self)} samples, "
                    f"{len(self.target_props)} target properties")
        
    def _load_and_index_data(self, path):
        """Load data files and build property index."""
        total = 0
        files = list(pathlib.Path(path).glob("*.pt"))
        
        logging.info(f"Loading and indexing {len(files)} data files...")
        for file_path in tqdm.tqdm(files, total=len(files)):
            obj = torch.load(file_path, map_location="cpu")
            selfies, assay_vals = obj["selfies"], obj["assay_vals"]
            self.data.append((selfies, assay_vals))
            
            # Index each example by the properties it contains (regardless of position)
            for local_idx, row in enumerate(assay_vals):
                global_idx = total + local_idx
                mask = row != self.pad_idx
                property_values = row[mask][1:-1].view(-1, 2)  # Extract property-value pairs
                
                # Extract property tokens and index them
                properties = set(int(prop) for prop, _ in property_values)
                for prop in properties:
                    if prop in self.target_props:
                        self.property_index[prop].append(global_idx)
            
            total += selfies.size(0)
            self.cumulative_lengths.append(total)
        
        logging.info(f"Loaded {total} examples")
        
    def _validate_targets(self):
        """Verify that we have examples for each targeted property."""
        missing_targets = []
        for prop in self.target_props:
            if not self.property_index[prop]:
                missing_targets.append(prop)
                
        if missing_targets:
            logging.warning(f"No examples found for {len(missing_targets)} out of {len(self.target_props)} missing property tokens: {missing_targets}")
            # Remove missing targets from sampling weights
            for missing in missing_targets:
                if missing in self.sampling_weights:
                    del self.sampling_weights[missing]
                self.target_props.discard(missing)
            
            # Re-normalize weights if needed
            if self.sampling_weights:
                total_weight = sum(self.sampling_weights.values())
                self.sampling_weights = {k: v/total_weight for k, v in self.sampling_weights.items()}
            else:
                raise ValueError("No valid property tokens found in the dataset")
    
    def __len__(self):
        return self.cumulative_lengths[-1]
    
    def _sample_property(self):
        """Sample a property token based on sampling weights."""
        props = list(self.sampling_weights.keys())
        weights = list(self.sampling_weights.values())
        return random.choices(props, weights=weights, k=1)[0]
        
    def _sample_index_and_property(self):
        """Sample an index and return both the index and the property it was sampled for."""
        prop = self._sample_property()
        candidates = self.property_index[prop]
        
        # Use frequency information to bias selection toward less frequently seen examples
        counts = self.sample_tracker.get_counts(candidates)
        
        # Convert counts to weights (inverse frequency)
        min_count = min(counts) if counts else 0
        max_count = max(counts) if counts else 0
        
        if min_count == max_count:
            # All candidates seen equally often, choose randomly
            idx = random.choice(candidates)
        else:
            # Weight by inverse frequency (plus small constant to avoid zeros)
            inv_weights = [1.0/(count - min_count + 1) for count in counts]
            total = sum(inv_weights)
            norm_weights = [w/total for w in inv_weights]
            
            idx = random.choices(candidates, weights=norm_weights, k=1)[0]
        
        return idx, prop

    def _extract_property_values(self, assay_vals):
        """Extract property-value pairs from assay values tensor."""
        mask = assay_vals != self.pad_idx
        pairs = assay_vals[mask][1:-1].view(-1, 2)  # Remove SEP and END tokens
        return [(int(prop), int(val)) for prop, val in pairs]
    
    def _rearrange_properties(self, properties, target_prop, target_pos):
        """
        Rearrange properties to ensure target_prop is at target_pos,
        and other properties are shuffled.
        """
        assert target_prop in [p for p, _ in properties], "Target property missing from selected example"

        # Filter out the target prop if it exists in the list
        other_props = [(p, v) for p, v in properties if p != target_prop]
        
        # Shuffle the other properties
        random.shuffle(other_props)
        
        # Check if we have enough properties to fill all positions
        result = []
        pos_counter = 0
        
        for pos in range(max(len(other_props) + 1, max(self.target_positions) + 1)):
            if pos == target_pos:
                # Insert our target property at the target position
                target_val = next((v for p, v in properties if p == target_prop), None)
                if target_val is None:
                    raise ValueError(f"Target property {target_prop} not found in properties list")
                result.append((target_prop, target_val))
            elif pos_counter < len(other_props):
                # Insert other properties
                result.append(other_props[pos_counter])
                pos_counter += 1
        
        return result
    
    def __getitem__(self, _):
        # Sample based on targeted property - get both index and the property used
        idx, target_prop = self._sample_index_and_property()
        
        # Update sample frequency
        self.sample_tracker.increment(idx)
        
        # Retrieve the corresponding data
        file_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        local_idx = idx - self.cumulative_lengths[file_idx]
        selfies = self.data[file_idx][0][local_idx]
        assay_vals = self.data[file_idx][1][local_idx]
        
        # Extract property-value pairs
        properties = self._extract_property_values(assay_vals)
        
        # Use the property that was used for sampling (guaranteed to be in properties)
        target_pos = random.choice(self.target_positions)
        
        # Rearrange properties to ensure target property is at target position
        rearranged = self._rearrange_properties(properties, target_prop, target_pos)
        
        # Flatten property-value pairs
        flat = [item for pair in rearranged for item in pair]
        
        # Truncate according to nprops
        flat = flat[:self.nprops*2]

        # Create output tensor
        device = selfies.device
        out = torch.tensor([self.sep_idx] + flat + [self.end_idx], device=device)
        
        # Pad if necessary
        max_len = self.nprops * 2 + 2
        if out.size(0) < max_len:
            out = F.pad(out, (0, max_len - out.size(0)), value=self.pad_idx)
            
        # Create teacher forcing input (shifted right)
        tch = torch.cat([torch.tensor([1], device=device), out[:-1]])
        
        return selfies, tch, out
    
    def sync_sample_counts(self):
        """Synchronize sample counts across GPU ranks."""
        self.sample_tracker.sync_across_ranks()
    
    def get_stats(self):
        """Return statistics about the dataset."""
        stats = {
            "total_examples": len(self),
            "target_counts": {str(prop): len(indices) 
                             for prop, indices in self.property_index.items()
                             if prop in self.target_props},
            "sampling_weights": {str(prop): weight 
                                for prop, weight in self.sampling_weights.items()}
        }
        return stats

    def set_sampling_weights(self, sampling_weights):
        """
        Update sampling weights and implicitly define the active sampling target set.

        Args:
            sampling_weights (dict): Mapping from property token to weight.
        """
        self.sampling_weights = sampling_weights.copy()
        
        # Remove any props not in the dataset
        valid_props = {prop for prop in sampling_weights if prop in self.property_index}
        if not valid_props:
            raise ValueError("None of the provided sampling weights correspond to known property tokens in the dataset.")
        
        # Re-normalize weights
        total_weight = sum(self.sampling_weights[prop] for prop in valid_props)
        self.sampling_weights = {prop: self.sampling_weights[prop] / total_weight for prop in valid_props}
        
        # Update target_props to reflect active sampling pool
        self.target_props = set(self.sampling_weights.keys())
        
        logging.info(f"Sampling weights updated. Active properties: {self.target_props}")

