from typing import Dict, List, Set, Tuple
from collections import defaultdict
import random

import logging
import torch
from torch.utils.data import Dataset
import pathlib
import tqdm
from typing import Optional, List, Tuple
from torch import Tensor
import torch.nn.functional as F

class InMemorySelfiesPropertiesValuesDataset(Dataset):
    
    def __init__(self, paths, tokenizer, nprops=5, assay_filter: List[int] = []):
        self.nprops = nprops
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.PAD_IDX
        self.assay_filter_tensor = torch.tensor(assay_filter, dtype=torch.long) if assay_filter else None

        # Each sample is (selfies_tensor, properties_tensor, values_tensor)
        self.samples: List[Tuple[Tensor, Tensor, Tensor]] = []

        # Handle both directory path, list of file paths, and single pt file
        if isinstance(paths, list):
            file_paths = [pathlib.Path(p) for p in paths]
        else:
            path = pathlib.Path(paths)
            if path.is_file() and path.suffix == '.pt':
                file_paths = [path]
            else:
                file_paths = list(path.glob("*.pt"))

        for file_path in tqdm.tqdm(file_paths, desc="Loading dataset into RAM"):
            file_data = torch.load(file_path, map_location="cpu", weights_only=True)
            
            # Your data format has these keys
            selfies_tensor = file_data["selfies"]
            properties_tensor = file_data["properties"]
            values_tensor = file_data["values"]

            # Process each sample in the batch
            for selfies, properties, values in zip(selfies_tensor, properties_tensor, values_tensor):
                # Filter out padding (-1 values)
                valid_mask = (properties != -1) & (values != -1)
                valid_properties = properties[valid_mask]
                valid_values = values[valid_mask]
                
                # Apply assay filter if provided
                if self.assay_filter_tensor is not None and self.assay_filter_tensor.numel() > 0:
                    filter_mask = torch.isin(valid_properties, self.assay_filter_tensor)
                    valid_properties = valid_properties[filter_mask]
                    valid_values = valid_values[filter_mask]
                
                # Only keep samples with at least one valid property-value pair
                if valid_properties.numel() > 0:
                    self.samples.append((selfies, valid_properties, valid_values))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        selfies, properties, values = self.samples[idx]
        
        # Process and pad properties and values
        processed_properties, processed_values = self._process_property_values(properties, values)
        
        # Create mask for non-padded elements
        mask = processed_properties != self.pad_idx
        
        # Normalize properties and values
        # normproperties = self.tokenizer.norm_properties(processed_properties, mask)
        # normvalues = self.tokenizer.norm_values(processed_values, mask)
        
        # return selfies, normproperties, normvalues, mask
        return selfies, processed_properties, processed_values, mask

    def _process_property_values(self, properties: Tensor, values: Tensor) -> Tuple[Tensor, Tensor]:
        """
        Process property and value tensors, randomly sampling nprops pairs.
        
        Args:
            properties: tensor of property indices
            values: tensor of corresponding values
            
        Returns:
            properties: [nprops] tensor of property tokens, padded if necessary
            values: [nprops] tensor of value tokens, padded if necessary
        """
        num_available = properties.size(0)
        
        # Randomly sample indices if we have more than nprops
        if num_available > self.nprops:
            # Random sampling without replacement
            perm = torch.randperm(num_available)[:self.nprops]
            sampled_properties = properties[perm]
            sampled_values = values[perm]
        else:
            sampled_properties = properties
            sampled_values = values
        
        # Create padded tensors
        actual_count = min(num_available, self.nprops)
        padded_properties = torch.full((self.nprops,), self.pad_idx, dtype=torch.long)
        padded_values = torch.full((self.nprops,), self.pad_idx, dtype=torch.long)
        
        if actual_count > 0:
            padded_properties[:actual_count] = sampled_properties[:actual_count]
            padded_values[:actual_count] = sampled_values[:actual_count]
        
        return padded_properties, padded_values