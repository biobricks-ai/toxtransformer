from typing import Dict, List, Set, Tuple
from collections import defaultdict
import random

import torch
from torch.utils.data import Dataset
import pathlib
import tqdm
import logging
from typing import Optional, List, Tuple
from torch import Tensor
import torch.nn.functional as F

class PreloadedTargetPropertyValuesDataset:
    """Preloaded version for more efficient access patterns."""
    
    def __init__(self, path, tokenizer):
        self.samples: List[Tuple[Tensor, Tensor, Tensor, int]] = []
        self.tokenizer = tokenizer
        self.pad_idx = 0  # Use 0 as pad index to avoid embedding lookup errors
        self.property_to_value_to_sample_idxs: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.sample_to_numprops: Dict[int, int] = defaultdict(int)

        sample_idx = 0
        for file_path in tqdm.tqdm(pathlib.Path(path).glob("*.pt"), desc="Preloading dataset"):
            file_data = torch.load(file_path, map_location="cpu", weights_only=True, mmap=True)
            
            # Handle new data format with separate tensors
            selfies_tensor = file_data["selfies"]
            properties_tensor = file_data["properties"]
            values_tensor = file_data["values"]

            for selfies, properties, values in zip(selfies_tensor, properties_tensor, values_tensor):
                # Filter out padding (-1 values)
                valid_mask = (properties != -1) & (values != -1)
                valid_properties = properties[valid_mask]
                valid_values = values[valid_mask]
                
                if len(valid_properties) == 0:
                    continue  # Skip empty samples
                
                num_properties = len(valid_properties)
                self.samples.append((selfies, valid_properties, valid_values, num_properties))
                
                # Map property values to sample indices
                for prop, value in zip(valid_properties.tolist(), valid_values.tolist()):
                    self.property_to_value_to_sample_idxs[prop][value].append(sample_idx)

                self.sample_to_numprops[sample_idx] = num_properties
                sample_idx += 1

class MultiTargetPropertyValuesWrapper(torch.utils.data.Dataset):
    """Wrapper that supports many (property, target_position) pairs."""

    def __init__(
        self,
        base: PreloadedTargetPropertyValuesDataset,
        prop_pos_pairs: List[Tuple[int, int]],
        nprops: int,
        nsamples: int = 200,
        minprops: int = 1,
        seed: Optional[int] = None,
    ):
        self.base = base
        self.tokenizer = base.tokenizer
        self.pad_idx = 0  # Use 0 instead of -1
        self.nprops = nprops
        self.nsamples = nsamples
        self.minprops = minprops

        if seed is not None:
            random.seed(seed)

        # Build an index list of (property_token, target_position, base_sample_idx)
        self.index_map: List[Tuple[int, int, int]] = []

        for prop_token, target_pos in prop_pos_pairs:
            # Collect all sample indices that contain this property
            value_to_idxs = base.property_to_value_to_sample_idxs.get(prop_token, {})
            all_idxs = []
            for v, idxs in value_to_idxs.items():
                for idx in idxs:
                    if base.sample_to_numprops.get(idx, 0) >= minprops:
                        all_idxs.append(idx)

            if not all_idxs:
                continue

            # Randomize and limit to nsamples per property-position pair
            selected = random.sample(all_idxs, min(nsamples, len(all_idxs)))

            # Append tuples in order so dataset iterates sequentially by pair
            for base_idx in selected:
                self.index_map.append((prop_token, int(target_pos), int(base_idx)))

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        prop_token, target_pos, base_idx = self.index_map[idx]
        bselfies, properties, values, num_properties = self.base.samples[base_idx]

        properties, values, mask = self._process_property_values(properties, values, prop_token, target_pos)

        return bselfies, properties, values, mask, num_properties

    def _process_property_values(self, properties: Tensor, values: Tensor, target_property_token: int, target_pos: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Shuffle non-target properties and place the target at requested position."""
        device = properties.device

        # Locate target and other properties
        target_mask = properties == target_property_token
        other_mask = ~target_mask

        target_properties = properties[target_mask]
        target_values = values[target_mask]
        other_properties = properties[other_mask]
        other_values = values[other_mask]

        if target_properties.size(0) == 0:
            raise ValueError("Target property token not found in properties tensor.")

        # Shuffle other properties
        if other_properties.size(0) > 0:
            perm = torch.randperm(other_properties.size(0))
            shuffled_other_properties = other_properties[perm]
            shuffled_other_values = other_values[perm]
            n_other = min(self.nprops - 1, shuffled_other_properties.size(0))
            selected_other_properties = shuffled_other_properties[:n_other]
            selected_other_values = shuffled_other_values[:n_other]
        else:
            selected_other_properties = torch.empty(0, dtype=properties.dtype, device=device)
            selected_other_values = torch.empty(0, dtype=values.dtype, device=device)
            n_other = 0

        # Insert target at target_pos
        target_pos = max(0, min(target_pos, self.nprops - 1))

        # Determine how many others to place to the left of the target
        left_space = target_pos
        left_count = min(n_other, left_space)
        
        # Split selected others into left and right portions
        left_properties = selected_other_properties[:left_count]
        left_values = selected_other_values[:left_count]
        right_properties = selected_other_properties[left_count:]
        right_values = selected_other_values[left_count:]

        # Build combined sequence: left, target, right
        combined_properties = torch.cat([left_properties, target_properties[0:1].to(device), right_properties], dim=0)
        combined_values = torch.cat([left_values, target_values[0:1].to(device), right_values], dim=0)

        actual_pairs = combined_properties.size(0)

        out_properties = torch.full((self.nprops,), self.pad_idx, dtype=torch.long, device=device)
        out_values = torch.full((self.nprops,), self.pad_idx, dtype=torch.long, device=device)
        mask = torch.zeros(self.nprops, dtype=torch.bool, device=device)

        if actual_pairs > 0:
            take = min(actual_pairs, self.nprops)
            out_properties[:take] = combined_properties[:take]
            out_values[:take] = combined_values[:take]
            mask[:take] = True

        return out_properties, out_values, mask


class MultiTaskPropertySamplesDataset(torch.utils.data.Dataset):
    """Flattened dataset that covers many (nprops, property_token, minprops) tasks."""

    def __init__(
        self,
        base: PreloadedTargetPropertyValuesDataset,
        task_tuples: List[Tuple[int, int, int]],
        nsamples: int = 200,
        max_nprops: Optional[int] = None,
        seed: Optional[int] = None,
    ):
        """
        Args:
            base: preloaded dataset
            task_tuples: sequence of (nprops, property_token, minprops)
            nsamples: max samples to draw per task
            max_nprops: global padded width; if None inferred from task_tuples
            seed: randomness seed
        """
        self.base = base
        self.tokenizer = base.tokenizer
        self.pad_idx = 0  # Use 0 instead of -1
        self.nsamples = nsamples

        if seed is not None:
            random.seed(seed)

        if max_nprops is None:
            max_nprops = max((t[0] for t in task_tuples), default=0)
        self.max_nprops = int(max_nprops)

        # Build flattened index: list of (base_idx, property_token, nprops, minprops)
        self.index_map: List[Tuple[int, int, int, int]] = []

        for nprops, prop_token, minprops in task_tuples:
            value_to_idxs = base.property_to_value_to_sample_idxs.get(prop_token, {})
            all_idxs = []
            for v, idxs in value_to_idxs.items():
                for idx in idxs:
                    if base.sample_to_numprops.get(idx, 0) >= minprops:
                        all_idxs.append(idx)

            if not all_idxs:
                continue

            random.shuffle(all_idxs)
            selected = all_idxs[:min(nsamples, len(all_idxs))]

            for base_idx in selected:
                self.index_map.append((int(base_idx), int(prop_token), int(nprops), int(minprops)))

        logging.info(f"MultiTaskPropertySamplesDataset: built index with {len(self.index_map)} items (max_nprops={self.max_nprops}, nsamples={self.nsamples})")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        base_idx, prop_token, nprops, minprops = self.index_map[idx]
        bselfies, properties, values, num_properties = self.base.samples[base_idx]

        properties, values, mask = self._process_property_values(properties, values, prop_token, nprops)

        # Pad to max_nprops if needed
        if properties.size(0) < self.max_nprops:
            props_pad = torch.full((self.max_nprops - properties.size(0),), self.pad_idx, dtype=torch.long)
            vals_pad = torch.full((self.max_nprops - values.size(0),), self.pad_idx, dtype=torch.long)
            mask_pad = torch.zeros(self.max_nprops - mask.size(0), dtype=torch.bool)
            properties = torch.cat([properties, props_pad], dim=0)
            values = torch.cat([values, vals_pad], dim=0)
            mask = torch.cat([mask, mask_pad], dim=0)

        # Return extra task metadata so caller can group/log by property/nprops/minprops
        return bselfies, properties, values, mask, num_properties, torch.tensor(prop_token, dtype=torch.long), torch.tensor(nprops, dtype=torch.long), torch.tensor(minprops, dtype=torch.long)

    def _process_property_values(self, properties: Tensor, values: Tensor, target_property_token: int, target_nprops: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Select up to target_nprops properties, place the target property at the last valid slot."""
        device = properties.device

        target_mask = properties == target_property_token
        other_mask = ~target_mask

        target_properties = properties[target_mask]
        target_values = values[target_mask]
        other_properties = properties[other_mask]
        other_values = values[other_mask]

        if target_properties.size(0) == 0:
            raise ValueError("Target property token not found in properties tensor.")

        if other_properties.size(0) > 0:
            perm = torch.randperm(other_properties.size(0))
            shuffled_other_properties = other_properties[perm]
            shuffled_other_values = other_values[perm]
            n_other = min(target_nprops - 1, shuffled_other_properties.size(0))
            selected_other_properties = shuffled_other_properties[:n_other]
            selected_other_values = shuffled_other_values[:n_other]
        else:
            selected_other_properties = torch.empty(0, dtype=properties.dtype, device=device)
            selected_other_values = torch.empty(0, dtype=values.dtype, device=device)
            n_other = 0

        # Place target at last valid position (target_nprops - 1)
        if n_other > 0:
            combined_properties = torch.cat([selected_other_properties, target_properties[0:1].to(device)], dim=0)
            combined_values = torch.cat([selected_other_values, target_values[0:1].to(device)], dim=0)
        else:
            combined_properties = target_properties[0:1].to(device)
            combined_values = target_values[0:1].to(device)

        actual_pairs = combined_properties.size(0)

        out_properties = torch.full((target_nprops,), self.pad_idx, dtype=torch.long)
        out_values = torch.full((target_nprops,), self.pad_idx, dtype=torch.long)
        mask = torch.zeros(target_nprops, dtype=torch.bool)

        if actual_pairs > 0:
            take = min(actual_pairs, target_nprops)
            out_properties[:take] = combined_properties[:take]
            out_values[:take] = combined_values[:take]
            mask[:take] = True

        return out_properties, out_values, mask