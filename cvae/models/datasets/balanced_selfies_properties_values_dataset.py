from typing import Dict, List, Set, Tuple
from collections import defaultdict
import random

import torch
from torch.utils.data import Dataset
import pathlib
import tqdm
from typing import Optional, List, Tuple
from torch import Tensor
import torch.nn.functional as F

class BalancedSelfiesPropertiesValuesDataset(Dataset):
    """
    A version of InMemorySelfiesPropertiesValuesDataset that ensures minimum sampling 
    for specified properties with both positive and negative values.
    
    This dataset guarantees that for each specified property, we sample at least N 
    examples with positive values and N examples with negative values, while keeping 
    the total number of samples minimal.
    """
    
    def __init__(self, paths, tokenizer, nprops=5, assay_filter: List[int] = [],
                 min_samples_per_property: int = 20, target_properties: List[int] = [],
                 positive_threshold: float = 0.5):
        self.nprops = nprops
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.PAD_IDX
        self.min_samples_per_property = min_samples_per_property
        self.target_properties = set(target_properties)
        self.positive_threshold = positive_threshold
        self.assay_filter_tensor = torch.tensor(assay_filter, dtype=torch.long) if assay_filter else None

        # Each sample is (selfies_tensor, reshaped_valid_tokens [N x 2])
        self.samples: List[Tuple[Tensor, Tensor]] = []
        
        # Index samples by property and value type for balanced sampling
        # property_value_index[property_id][value_type] = [sample_indices]
        # value_type: 'positive' or 'negative'
        # Fixed: Use regular dict instead of defaultdict with lambda to avoid serialization issues
        self.property_value_index: Dict[int, Dict[str, List[int]]] = {}

        # Handle both directory path, list of file paths, and single pt file
        if isinstance(paths, list):
            file_paths = [pathlib.Path(p) for p in paths]
        else:
            path = pathlib.Path(paths)
            if path.is_file() and path.suffix == '.pt':
                file_paths = [path]
            else:
                file_paths = list(path.glob("*.pt"))

        print(f"🔍 Building balanced dataset with min {min_samples_per_property} samples per property type")
        print(f"🎯 Target properties: {target_properties}")

        # First pass: load all samples and build index
        for file_path in tqdm.tqdm(file_paths, desc="Loading dataset into RAM"):
            file_data = torch.load(file_path, map_location="cpu")
            selfies_list = file_data["selfies"]
            assay_vals_list = file_data["assay_vals"]

            for selfies, assay_vals in zip(selfies_list, assay_vals_list):
                mask = assay_vals != self.pad_idx
                valid_tokens = assay_vals[mask][1:-1]  # Remove SEP and END tokens

                if self.assay_filter_tensor is not None and self.assay_filter_tensor.numel() > 0:
                    assay_ids = valid_tokens[::2]  # Property tokens at even indices
                    assay_mask = torch.isin(assay_ids, self.assay_filter_tensor)
                    expanded_mask = torch.zeros_like(valid_tokens, dtype=torch.bool)
                    expanded_mask[::2] = assay_mask  # Properties
                    expanded_mask[1::2] = assay_mask  # Values
                    valid_tokens = valid_tokens[expanded_mask]

                if valid_tokens.numel() >= 2:
                    reshaped = valid_tokens.view(-1, 2).contiguous()  # [num_pairs, 2]
                    sample_idx = len(self.samples)
                    self.samples.append((selfies, reshaped))
                    
                    # Index this sample by its properties and value types
                    self._index_sample(sample_idx, reshaped)

        print(f"📊 Loaded {len(self.samples)} total samples")
        self._print_balance_statistics()
        
        # Create balanced sample indices
        self.balanced_indices = self._create_balanced_indices()
        print(f"🎯 Selected {len(self.balanced_indices)} samples for balanced dataset")

    def _index_sample(self, sample_idx: int, property_value_pairs: Tensor):
        """Index a sample by its properties and value types."""
        properties = property_value_pairs[:, 0]  # Property tokens
        values = property_value_pairs[:, 1]      # Value tokens
        
        # Normalize values to get actual values (0-indexed)
        norm_values = values - self.tokenizer.properties_offset
        
        for prop_token, norm_value in zip(properties, norm_values):
            # Convert property token to property ID
            prop_id = (prop_token - self.tokenizer.selfies_offset).item()
            
            # Only index target properties
            if prop_id in self.target_properties:
                # Initialize nested dict if needed
                if prop_id not in self.property_value_index:
                    self.property_value_index[prop_id] = {'positive': [], 'negative': []}
                
                # Determine if this is a positive or negative value
                value_type = 'positive' if norm_value.item() >= self.positive_threshold else 'negative'
                self.property_value_index[prop_id][value_type].append(sample_idx)

    def _print_balance_statistics(self):
        """Print statistics about the balance of properties."""
        print("\n📈 Property balance statistics:")
        for prop_id in sorted(self.target_properties):
            if prop_id in self.property_value_index:
                pos_count = len(self.property_value_index[prop_id]['positive'])
                neg_count = len(self.property_value_index[prop_id]['negative'])
                print(f"   Property {prop_id}: {pos_count} positive, {neg_count} negative")
            else:
                print(f"   Property {prop_id}: 0 positive, 0 negative")

    def _create_balanced_indices(self) -> List[int]:
        """Create a list of sample indices that satisfies balance requirements."""
        selected_indices = set()
        
        # For each target property, ensure minimum samples for both positive and negative
        for prop_id in self.target_properties:
            if prop_id not in self.property_value_index:
                print(f"⚠️  Warning: Property {prop_id} has no samples")
                continue
                
            pos_samples = self.property_value_index[prop_id]['positive']
            neg_samples = self.property_value_index[prop_id]['negative']
            
            # Sample minimum required for each type
            pos_selected = random.sample(
                pos_samples, 
                min(self.min_samples_per_property, len(pos_samples))
            )
            neg_selected = random.sample(
                neg_samples, 
                min(self.min_samples_per_property, len(neg_samples))
            )
            
            selected_indices.update(pos_selected)
            selected_indices.update(neg_selected)
            
            print(f"🎯 Property {prop_id}: selected {len(pos_selected)} positive, {len(neg_selected)} negative")
        
        # Add some additional random samples to reach a reasonable dataset size
        # but avoid making it too large
        remaining_samples = set(range(len(self.samples))) - selected_indices
        additional_count = min(len(remaining_samples), len(selected_indices) // 2)
        if additional_count > 0:
            additional_selected = random.sample(list(remaining_samples), additional_count)
            selected_indices.update(additional_selected)
        
        return sorted(list(selected_indices))

    def __len__(self):
        return len(self.balanced_indices)

    def __getitem__(self, idx):
        # Map to actual sample index
        actual_idx = self.balanced_indices[idx]
        selfies, property_value_pairs = self.samples[actual_idx]
        
        properties, values = self._process_property_values(property_value_pairs)
        mask = properties != self.pad_idx
        normproperties = self.tokenizer.norm_properties(properties, mask)
        normvalues = self.tokenizer.norm_values(values, mask)
        return selfies, normproperties, normvalues, mask

    def _process_property_values(self, property_value_pairs: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Process property-value pairs into separate properties and values tensors.
        
        Args:
            property_value_pairs: [num_pairs, 2] tensor where each row is [property, value]
            
        Returns:
            properties: [nprops] tensor of property tokens, padded if necessary
            values: [nprops] tensor of value tokens, padded if necessary
        """
        # Randomly shuffle and truncate to nprops
        perm = torch.randperm(property_value_pairs.size(0))
        shuffled_pairs = property_value_pairs[perm]
        truncated_pairs = shuffled_pairs[:self.nprops]
        
        # Split into properties and values
        actual_pairs = truncated_pairs.size(0)
        
        # Create padded tensors
        properties = torch.full((self.nprops,), self.pad_idx, dtype=torch.long)
        values = torch.full((self.nprops,), self.pad_idx, dtype=torch.long)
        
        if actual_pairs > 0:
            properties[:actual_pairs] = truncated_pairs[:, 0]  # Property tokens
            values[:actual_pairs] = truncated_pairs[:, 1]      # Value tokens
        
        return properties, values

    def get_property_statistics(self) -> Dict:
        """Get statistics about property distribution in the balanced dataset."""
        stats = {}
        property_counts = {}
        
        for prop_id in self.target_properties:
            property_counts[prop_id] = {'positive': 0, 'negative': 0}
        
        for idx in self.balanced_indices:
            _, property_value_pairs = self.samples[idx]
            properties = property_value_pairs[:, 0]
            values = property_value_pairs[:, 1]
            norm_values = values - self.tokenizer.properties_offset
            
            for prop_token, norm_value in zip(properties, norm_values):
                prop_id = (prop_token - self.tokenizer.selfies_offset).item()
                if prop_id in self.target_properties:
                    value_type = 'positive' if norm_value.item() >= self.positive_threshold else 'negative'
                    property_counts[prop_id][value_type] += 1
        
        return property_counts
