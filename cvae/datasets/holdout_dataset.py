"""
Holdout dataset for evaluation.

Loads tensor files from bootstrap splits and unfolds into individual
(selfies, property, value) samples. Supports both independent evaluation
and autoregressive evaluation by tracking compound membership.
"""
import logging
import pathlib
from typing import Dict, List, Optional, Tuple

import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class HoldoutDataset(Dataset):
    """
    Dataset that loads holdout tensor files and unfolds into individual samples.

    Each tensor file contains compounds with multiple (property, value) pairs.
    This dataset unfolds them so each sample is a single (selfies, property, value).

    For autoregressive evaluation, tracks which properties belong to the same
    compound via compound_id, allowing context lookup.

    Args:
        split_id: Bootstrap split index (0-4)
        tensor_base: Base path for tensor files
        rank: Distributed rank for logging control
        include_compound_properties: If True, include all properties for each
            compound (needed for autoregressive evaluation)
    """

    def __init__(
        self,
        split_id: int,
        tensor_base: str = 'cache/build_tensordataset/bootstrap',
        rank: int = 0,
        include_compound_properties: bool = False,
    ):
        self.split_id = split_id
        self.tensor_base = tensor_base
        self.include_compound_properties = include_compound_properties

        # Storage
        self.samples: List[Dict] = []

        # For autoregressive: map compound_id -> all (property, value) pairs
        self.compound_properties: Dict[int, List[Tuple[int, int]]] = {}

        # Load data
        self._load_tensor_files(rank)

    def _load_tensor_files(self, rank: int):
        """Load and unfold all tensor files for this split."""
        test_dir = pathlib.Path(f"{self.tensor_base}/split_{self.split_id}/test")
        tensor_files = sorted(test_dir.glob("*.pt"))

        logging.info(f"[Rank {rank}] Loading {len(tensor_files)} tensor files for split {self.split_id}")

        compound_id = 0

        for tensor_file in tqdm(
            tensor_files,
            desc=f"Loading split {self.split_id}",
            disable=rank != 0,
            unit="file",
            dynamic_ncols=True,
        ):
            data = torch.load(tensor_file, map_location="cpu", weights_only=True)

            selfies_tensor = data["selfies"]      # [N, seq_len]
            properties_tensor = data["properties"]  # [N, max_props]
            values_tensor = data["values"]          # [N, max_props]

            # Unfold each compound
            for selfies, properties, values in zip(
                selfies_tensor, properties_tensor, values_tensor
            ):
                # Filter padding (-1 values)
                valid_mask = (properties != -1) & (values != -1)
                valid_properties = properties[valid_mask].tolist()
                valid_values = values[valid_mask].tolist()

                if not valid_properties:
                    continue

                # Store compound's properties for autoregressive lookup
                if self.include_compound_properties:
                    self.compound_properties[compound_id] = list(
                        zip(valid_properties, valid_values)
                    )

                # Convert selfies tensor to string identifier for joining tables
                selfies_tokens = ' '.join(str(x) for x in selfies.tolist())

                # Create one sample per (property, value) pair
                for prop, val in zip(valid_properties, valid_values):
                    sample = {
                        'selfies': selfies,
                        'selfies_tokens': selfies_tokens,
                        'property': prop,
                        'value': val,
                        'compound_id': compound_id,
                    }
                    self.samples.append(sample)

                compound_id += 1

        logging.info(
            f"[Rank {rank}] Loaded {len(self.samples):,} samples "
            f"from {compound_id:,} compounds in split {self.split_id}"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        sample = self.samples[idx]

        if self.include_compound_properties:
            # Include all properties for this compound (for autoregressive)
            compound_id = sample['compound_id']
            sample = sample.copy()
            sample['compound_properties'] = self.compound_properties[compound_id]

        return sample

    def get_compound_properties(self, compound_id: int) -> List[Tuple[int, int]]:
        """Get all (property, value) pairs for a compound."""
        return self.compound_properties.get(compound_id, [])


def collate_nprops1(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """
    Collate batch for nprops=1 evaluation (no context).

    Returns dict with:
        selfies: [batch, seq_len]
        properties: [batch, 1]
        values: [batch, 1]
        mask: [batch, 1]
        true_values: [batch]
        property_tokens: [batch]
        selfies_tokens: List[str] - for joining with context tables
    """
    return {
        'selfies': torch.stack([item['selfies'] for item in batch]),
        'properties': torch.tensor(
            [[item['property']] for item in batch], dtype=torch.long
        ),
        'values': torch.tensor(
            [[item['value']] for item in batch], dtype=torch.long
        ),
        'mask': torch.ones((len(batch), 1), dtype=torch.bool),
        'true_values': torch.tensor(
            [item['value'] for item in batch], dtype=torch.long
        ),
        'property_tokens': torch.tensor(
            [item['property'] for item in batch], dtype=torch.long
        ),
        'selfies_tokens': [item['selfies_tokens'] for item in batch],
    }


def collate_autoregressive(batch: List[Dict]) -> List[Dict]:
    """
    Collate batch for autoregressive evaluation.

    Keeps samples as list since we need compound_properties for context.
    """
    return batch
