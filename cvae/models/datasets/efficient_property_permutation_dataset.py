import torch
import torch.utils.data
import logging
import random
from typing import List, Tuple, Set
from torch import Tensor
from tqdm import tqdm

class EfficientPropertyPermutationDataset(torch.utils.data.Dataset):
    """Simple dataset that puts each target property at each possible position.
    
    For each sample with target properties, creates variations where each target
    property appears at each position (0 to nprops-1), filling remaining positions
    with other available properties up to max_permutations_per_target_position.
    """

    def __init__(
        self,
        base: 'PreloadedTargetPropertyValuesDataset',
        target_properties: Set[int],
        nprops: int,
        max_permutations_per_target_position: int = 5,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 42,
    ):
        """
        Args:
            base: preloaded dataset  
            target_properties: set of property tokens to consider as targets
            nprops: number of properties per sample
            max_permutations_per_target_position: max variations per (target_prop, position) pair
            rank: current process rank for distributed training
            world_size: total number of processes
            seed: random seed for reproducible sampling
        """
        self.base = base
        self.tokenizer = base.tokenizer
        self.pad_idx = self.tokenizer.PAD_IDX
        self.target_properties = target_properties
        self.nprops = nprops
        self.max_permutations = max_permutations_per_target_position
        self.rank = rank
        self.world_size = world_size
        self.rng = random.Random(seed)
        
        # Build sample variations: (base_idx, target_prop, position, other_props_sample)
        self.variations: List[Tuple[int, int, int, List[Tuple[int, int]]]] = []
        self._build_variations()
        
        # Distribute variations across ranks
        total_variations = len(self.variations)
        variations_per_rank = total_variations // world_size
        remaining_variations = total_variations % world_size
        
        # Ranks 0 to remaining_variations-1 get one extra variation
        if rank < remaining_variations:
            self.rank_length = variations_per_rank + 1
            start_idx = rank * (variations_per_rank + 1)
        else:
            self.rank_length = variations_per_rank
            start_idx = remaining_variations * (variations_per_rank + 1) + (rank - remaining_variations) * variations_per_rank
        
        # Extract this rank's variations
        self.rank_variations = self.variations[start_idx:start_idx + self.rank_length]
        
        logging.info(f"EfficientPropertyPermutationDataset: rank {rank}/{world_size} "
                    f"has {self.rank_length} variations (total: {total_variations})")

    def _build_variations(self):
        """Build all variations for samples with target properties."""
        for base_idx in tqdm(range(len(self.base.samples)), desc="Building variations"):
            _, reshaped, total_properties = self.base.samples[base_idx]
            
            if reshaped.size(0) == 0:
                continue
            
            # Find target properties in this sample
            available_props = [(reshaped[i, 0].item(), reshaped[i, 1].item()) 
                             for i in range(reshaped.size(0))]
            target_props_in_sample = [(prop, val) for prop, val in available_props 
                                    if prop in self.target_properties]
            
            if not target_props_in_sample:
                continue
            
            # Get non-target properties for filling other positions
            non_target_props = [(prop, val) for prop, val in available_props 
                              if prop not in self.target_properties]
            
            # For each target property in this sample
            for target_prop, target_val in target_props_in_sample:
                # For each possible position (0 to nprops-1)
                for position in range(self.nprops):
                    # Generate up to max_permutations variations for this (target_prop, position)
                    variations_count = 0
                    
                    # If we have enough non-target props, sample different combinations
                    if len(non_target_props) >= self.nprops - 1:
                        # Generate multiple random samples of other properties
                        seen_combinations = set()
                        attempts = 0
                        max_attempts = self.max_permutations * 10  # Avoid infinite loops
                        
                        while variations_count < self.max_permutations and attempts < max_attempts:
                            # Sample nprops-1 other properties to fill remaining positions
                            other_props = self.rng.sample(non_target_props, self.nprops - 1)
                            other_props_tuple = tuple(sorted(other_props))  # For deduplication
                            
                            if other_props_tuple not in seen_combinations:
                                seen_combinations.add(other_props_tuple)
                                self.variations.append((base_idx, target_prop, position, other_props))
                                variations_count += 1
                            
                            attempts += 1
                    else:
                        # Not enough non-target props, just create one variation with all available
                        other_props = non_target_props.copy()
                        self.variations.append((base_idx, target_prop, position, other_props))

    def __len__(self):
        return self.rank_length

    def __getitem__(self, idx):
        if idx >= self.rank_length:
            raise IndexError(f"Index {idx} out of range for rank {self.rank}")
        
        base_idx, target_prop, target_position, other_props = self.rank_variations[idx]
        
        # Get base sample data
        bselfies, reshaped, total_properties = self.base.samples[base_idx]
        
        # Find the target property's value in the original sample
        target_val = None
        for i in range(reshaped.size(0)):
            if reshaped[i, 0].item() == target_prop:
                target_val = reshaped[i, 1].item()
                break
        
        if target_val is None:
            raise ValueError(f"Target property {target_prop} not found in sample {base_idx}")
        
        # Build the property sequence
        properties = torch.full((self.nprops,), self.pad_idx, dtype=torch.long)
        values = torch.full((self.nprops,), self.pad_idx, dtype=torch.long)
        mask = torch.zeros(self.nprops, dtype=torch.bool)
        
        # Place target property at specified position
        properties[target_position] = target_prop
        values[target_position] = target_val
        mask[target_position] = True
        
        # Fill other positions with other_props
        other_positions = [i for i in range(self.nprops) if i != target_position]
        for i, (prop, val) in enumerate(other_props):
            if i < len(other_positions):
                pos = other_positions[i]
                properties[pos] = prop
                values[pos] = val
                mask[pos] = True
        
        # Normalize properties and values
        normproperties = self.base.tokenizer.norm_properties(properties, mask)
        normvalues = self.base.tokenizer.norm_values(values, mask)
        
        return (
            bselfies,
            normproperties,
            normvalues,
            mask,
            torch.tensor(total_properties, dtype=torch.long),
        )