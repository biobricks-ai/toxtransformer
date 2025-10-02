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
        self.samples: List[Tuple[Tensor, Tensor, int]] = []
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.PAD_IDX
        self.property_to_value_to_sample_idxs: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.sample_to_numprops: Dict[int, int] = defaultdict(int)

        sample_idx = 0
        for file_path in tqdm.tqdm(pathlib.Path(path).glob("*.pt"), desc="Preloading dataset"):
            file_data = torch.load(file_path, map_location="cpu")
            selfies_list = file_data["selfies"]
            assay_vals_list = file_data["assay_vals"]

            for selfies, assay_vals in zip(selfies_list, assay_vals_list):
                mask = assay_vals != self.pad_idx
                valid_tokens = assay_vals[mask][1:-1]
                if valid_tokens.numel() >= 2:
                    reshaped = valid_tokens.view(-1, 2).contiguous()
                    num_properties = reshaped.size(0)
                    self.samples.append((selfies, reshaped, num_properties))
                    
                    # Map property values to sample indices
                    for prop, value in zip(reshaped[:, 0].tolist(), reshaped[:, 1].tolist()):
                        self.property_to_value_to_sample_idxs[prop][value].append(sample_idx)

                    self.sample_to_numprops[sample_idx] = num_properties
                    sample_idx += 1


class TargetPropertyValuesWrapper(torch.utils.data.Dataset):
    """Wrapper that creates balanced target property datasets from preloaded data."""

    def __init__(self, base: PreloadedTargetPropertyValuesDataset, target_property_token: int, nprops: int, nsamples: int, minprops: int):
        self.base = base
        self.tokenizer = base.tokenizer
        self.nprops = nprops
        self.pad_idx = self.tokenizer.PAD_IDX
        self.target_property_token = target_property_token

        self.base_samples = base.samples
        
        # Get all unique values for the target property
        value_to_idxs = base.property_to_value_to_sample_idxs[target_property_token]
        
        # For each value, collect sample indices with at least minprops properties
        val_selected = defaultdict(list)
        for value, idxs in value_to_idxs.items():
            for idx in idxs:
                if base.sample_to_numprops[idx] >= minprops:
                    val_selected[value].append(idx)

        # Compute the minimum number of samples available per value
        min_selected_lengths = min((len(v) for v in val_selected.values()), default=0)
        per_value_samples = min(nsamples, min_selected_lengths)

        # Build the final sample indices list: for each value, randomly select per_value_samples indices
        self.samples: List[int] = []
        for value, idxs in val_selected.items():
            random.shuffle(idxs)  # Randomize before selecting
            self.samples.extend(idxs[:per_value_samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_idx = self.samples[idx]
        bselfies, reshaped, num_properties = self.base_samples[base_idx]
        properties, values, mask = self._process_property_values(reshaped)
        normproperties = self.base.tokenizer.norm_properties(properties, mask)
        normvalues = self.base.tokenizer.norm_values(values, mask)
        return bselfies, normproperties, normvalues, mask, num_properties

    def _process_property_values(self, property_value_pairs: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Process property-value pairs ensuring target property is in the last position.
        
        Args:
            property_value_pairs: [num_pairs, 2] tensor where each row is [property, value]
            
        Returns:
            properties: [nprops] tensor of property tokens, padded if necessary
            values: [nprops] tensor of value tokens, padded if necessary  
            mask: [nprops] tensor indicating valid positions
        """
        device = property_value_pairs.device

        # Split out the target row and other rows
        target_rows = property_value_pairs[property_value_pairs[:, 0] == self.target_property_token]
        other_rows = property_value_pairs[property_value_pairs[:, 0] != self.target_property_token]

        if target_rows.size(0) == 0:
            raise ValueError("Target property token not found in property_value_pairs tensor.")

        # Shuffle other rows and take up to (nprops - 1)
        if other_rows.size(0) > 0:
            perm = torch.randperm(other_rows.size(0))
            shuffled_others = other_rows[perm]
            n_other = min(self.nprops - 1, shuffled_others.size(0))
            selected_others = shuffled_others[:n_other]
        else:
            selected_others = torch.empty(0, 2, dtype=property_value_pairs.dtype, device=device)
            n_other = 0

        # Combine: other properties first, then target property last
        if n_other > 0:
            combined_pairs = torch.cat([selected_others, target_rows[0:1]], dim=0)
        else:
            combined_pairs = target_rows[0:1]
        
        actual_pairs = combined_pairs.size(0)
        
        # Create padded tensors
        properties = torch.full((self.nprops,), self.pad_idx, dtype=torch.long, device=device)
        values = torch.full((self.nprops,), self.pad_idx, dtype=torch.long, device=device)
        mask = torch.zeros(self.nprops, dtype=torch.bool, device=device)
        
        if actual_pairs > 0:
            properties[:actual_pairs] = combined_pairs[:, 0]  # Property tokens
            values[:actual_pairs] = combined_pairs[:, 1]      # Value tokens
            mask[:actual_pairs] = True
        
        return properties, values, mask


class MultiTargetPropertyValuesWrapper(torch.utils.data.Dataset):
    """Wrapper that supports many (property, target_position) pairs.

    For each (property_token, target_position) pair this dataset selects up to
    `nsamples` sample indices from `base` that contain the property and have
    at least `minprops` properties. The dataset iterates sequentially through
    the provided pairs, producing one example per selected sample per pair.

    Each returned sample is processed so that the target property for that
    step is placed at `target_position` (0-based) within the `nprops` slots.
    Other properties are shuffled before insertion to provide randomness.
    The returned tensors are normalized via the base tokenizer to match the
    model's expected input (properties, values, mask).
    """

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
        self.pad_idx = self.tokenizer.PAD_IDX
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
        bselfies, reshaped, num_properties = self.base.samples[base_idx]

        properties, values, mask = self._process_property_values(reshaped, prop_token, target_pos)

        # Normalize using tokenizer to be consistent with other wrappers
        normproperties = self.base.tokenizer.norm_properties(properties, mask)
        normvalues = self.base.tokenizer.norm_values(values, mask)

        return bselfies, normproperties, normvalues, mask, num_properties

    def _process_property_values(self, property_value_pairs: Tensor, target_property_token: int, target_pos: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Shuffle non-target property-value pairs and place the target pair at
        the requested `target_pos` (0-based) within the `nprops` window.

        Behavior:
        - Find the first occurrence of the target property in `property_value_pairs`.
        - Shuffle the other property rows and select up to `nprops-1` of them.
        - Insert the target row at `target_pos` among the selected rows.
        - Pad properties/values to length `nprops` using PAD_IDX and build mask.
        """
        device = property_value_pairs.device

        # Locate target and other rows
        target_rows = property_value_pairs[property_value_pairs[:, 0] == target_property_token]
        other_rows = property_value_pairs[property_value_pairs[:, 0] != target_property_token]

        if target_rows.size(0) == 0:
            raise ValueError("Target property token not found in property_value_pairs tensor.")

        # Shuffle other rows
        if other_rows.size(0) > 0:
            perm = torch.randperm(other_rows.size(0))
            shuffled_others = other_rows[perm]
            n_other = min(self.nprops - 1, shuffled_others.size(0))
            selected_others = shuffled_others[:n_other]
        else:
            selected_others = torch.empty(0, 2, dtype=property_value_pairs.dtype, device=device)
            n_other = 0

        # We'll insert the target row at target_pos (clamped)
        target_pos = max(0, min(target_pos, self.nprops - 1))

        # Determine how many others to place to the left of the target
        left_space = target_pos
        left_count = min(n_other, left_space)
        # Split selected others into left and right portions
        left = selected_others[:left_count]
        right = selected_others[left_count:]

        # Build combined sequence: left, target, right
        combined = torch.cat([left, target_rows[0:1].to(device), right], dim=0)

        actual_pairs = combined.size(0)

        properties = torch.full((self.nprops,), self.pad_idx, dtype=torch.long, device=device)
        values = torch.full((self.nprops,), self.pad_idx, dtype=torch.long, device=device)
        mask = torch.zeros(self.nprops, dtype=torch.bool, device=device)

        if actual_pairs > 0:
            # If combined is longer than nprops (shouldn't happen), truncate
            take = min(actual_pairs, self.nprops)
            properties[:take] = combined[:take, 0]
            values[:take] = combined[:take, 1]
            mask[:take] = True

        return properties, values, mask


class MultiTaskPropertySamplesDataset(torch.utils.data.Dataset):
    """Flattened dataset that covers many (nprops, property_token, minprops) tasks.

    Each item corresponds to one sample drawn for a particular task. All returned
    property/value tensors are padded to `max_nprops` so batches have consistent
    shapes and can be collated by DataLoader.
    """

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
        self.pad_idx = self.tokenizer.PAD_IDX
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
        bselfies, reshaped, num_properties = self.base.samples[base_idx]

        properties, values, mask = self._process_property_values(reshaped, prop_token, nprops)

        # Pad to max_nprops if needed
        if properties.size(0) < self.max_nprops:
            props_pad = torch.full((self.max_nprops - properties.size(0),), self.pad_idx, dtype=torch.long)
            vals_pad = torch.full((self.max_nprops - values.size(0),), self.pad_idx, dtype=torch.long)
            mask_pad = torch.zeros(self.max_nprops - mask.size(0), dtype=torch.bool)
            properties = torch.cat([properties, props_pad], dim=0)
            values = torch.cat([values, vals_pad], dim=0)
            mask = torch.cat([mask, mask_pad], dim=0)

        normproperties = self.base.tokenizer.norm_properties(properties, mask)
        normvalues = self.base.tokenizer.norm_values(values, mask)

        # Return extra task metadata so caller can group/log by property/nprops/minprops
        return bselfies, normproperties, normvalues, mask, num_properties, torch.tensor(prop_token, dtype=torch.long), torch.tensor(nprops, dtype=torch.long), torch.tensor(minprops, dtype=torch.long)

    def _process_property_values(self, property_value_pairs: Tensor, target_property_token: int, target_nprops: int) -> Tuple[Tensor, Tensor, Tensor]:
        """Select up to target_nprops properties, place the target property at the last valid slot.

        Returns tensors sized target_nprops (not padded to max here).
        """
        device = property_value_pairs.device

        target_rows = property_value_pairs[property_value_pairs[:, 0] == target_property_token]
        other_rows = property_value_pairs[property_value_pairs[:, 0] != target_property_token]

        if target_rows.size(0) == 0:
            raise ValueError("Target property token not found in property_value_pairs tensor.")

        if other_rows.size(0) > 0:
            perm = torch.randperm(other_rows.size(0))
            shuffled_others = other_rows[perm]
            n_other = min(target_nprops - 1, shuffled_others.size(0))
            selected_others = shuffled_others[:n_other]
        else:
            selected_others = torch.empty(0, 2, dtype=property_value_pairs.dtype, device=device)
            n_other = 0

        # Place target at last valid position (target_nprops - 1)
        if n_other > 0:
            combined_pairs = torch.cat([selected_others, target_rows[0:1].to(device)], dim=0)
        else:
            combined_pairs = target_rows[0:1].to(device)

        actual_pairs = combined_pairs.size(0)

        properties = torch.full((target_nprops,), self.pad_idx, dtype=torch.long)
        values = torch.full((target_nprops,), self.pad_idx, dtype=torch.long)
        mask = torch.zeros(target_nprops, dtype=torch.bool)

        if actual_pairs > 0:
            take = min(actual_pairs, target_nprops)
            properties[:take] = combined_pairs[:take, 0]
            values[:take] = combined_pairs[:take, 1]
            mask[:take] = True

        return properties, values, mask
