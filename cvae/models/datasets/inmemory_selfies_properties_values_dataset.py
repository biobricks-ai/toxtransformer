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

class InMemorySelfiesPropertiesValuesDataset(Dataset):

    @staticmethod
    def from_cache_or_create(paths, tokenizer, nprops=5, assay_filter: List[int] = [], cache_path=None):
        """
        Creates dataset from cache if it exists, otherwise builds and caches it.
        """
        if cache_path is not None:
            cache_path = pathlib.Path(cache_path)
            if cache_path.exists():
                print(f"✅ Loading dataset from cache: {cache_path}")
                dataset = InMemorySelfiesPropertiesValuesDataset.__new__(InMemorySelfiesPropertiesValuesDataset)
                dataset.samples = torch.load(cache_path, map_location="cpu")
                dataset.nprops = nprops
                dataset.tokenizer = tokenizer
                dataset.pad_idx = tokenizer.PAD_IDX
                return dataset

        # Otherwise, create and cache it
        dataset = InMemorySelfiesPropertiesValuesDataset(paths, tokenizer, nprops, assay_filter)
        if cache_path is not None:
            print(f"💾 Caching dataset to: {cache_path}")
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(dataset.samples, cache_path)

        return dataset
    
    def __init__(self, paths, tokenizer, nprops=5, assay_filter: List[int] = []):
        self.nprops = nprops
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.PAD_IDX
        self.assay_filter_tensor = torch.tensor(assay_filter, dtype=torch.long) if assay_filter else None

        # Each sample is (selfies_tensor, reshaped_valid_tokens [N x 2])
        self.samples: List[Tuple[Tensor, Tensor]] = []

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
                    self.samples.append((selfies, reshaped))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        selfies, property_value_pairs = self.samples[idx]
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