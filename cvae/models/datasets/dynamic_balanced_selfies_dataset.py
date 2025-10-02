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

# TODO go back to last version
class DynamicBalancedSelfiesDataset(Dataset):
    """
    A dataset that dynamically balances property-value pair sampling during __getitem__.
    
    This dataset loads all samples and then dynamically samples during training to ensure
    balanced exposure of specified property-value pairs. It ignores the provided index
    and instead performs intelligent sampling to maintain balance.
    """
    
    def __init__(self, paths, tokenizer, nprops=5, assay_filter: List[int] = [],
                 target_properties: List[int] = [], balanced_ratio: float = 0.8):
        self.nprops = nprops
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.PAD_IDX
        self.target_properties = set(target_properties)
        self.assay_filter_tensor = torch.tensor(assay_filter, dtype=torch.long) if assay_filter else None
        self.balanced_ratio = balanced_ratio  # Fraction of sampling that should be balanced

        # Each sample is (selfies_tensor, reshaped_valid_tokens [N x 2])
        self.samples: List[Tuple[Tensor, Tensor]] = []
        
        # Index: property_id -> list of sample indices that contain this property
        self.property_sample_index: Dict[int, List[int]] = defaultdict(list)
        
        # Index: (property_id, value_token) -> list of sample indices
        self.property_value_index: Dict[Tuple[int, int], List[int]] = defaultdict(list)

        # Epoch tracking for full coverage
        self.current_epoch = 0
        self.samples_seen_this_epoch: Set[int] = set()
        self.unseen_samples_queue: List[int] = []

        # Handle paths
        if isinstance(paths, list):
            file_paths = [pathlib.Path(p) for p in paths]
        else:
            path = pathlib.Path(paths)
            if path.is_file() and path.suffix == '.pt':
                file_paths = [path]
            else:
                file_paths = list(path.glob("*.pt"))

        logging.info(f"🔍 Loading dataset for dynamic balanced sampling")
        logging.info(f"🎯 Target properties: {target_properties}")
        logging.info(f"📁 Processing {len(file_paths)} files")

        # Load all samples and build indices
        self._load_and_index(file_paths)
        
        logging.info(f"📊 Loaded {len(self.samples)} total samples")
        self._print_index_statistics()
        
        # Initialize sampling counters for balanced sampling
        self._init_sampling_counters()
        
        # Initialize epoch tracking
        self._reset_epoch()

    def _load_and_index(self, file_paths: List[pathlib.Path]):
        """Load and index files sequentially with progress bar."""
        total_samples = 0
        failed_files = 0
        
        # Use tqdm for progress tracking
        with tqdm.tqdm(file_paths, desc="Loading dataset", unit="files") as pbar:
            for file_path in pbar:
                try:
                    file_data = torch.load(file_path, map_location="cpu", weights_only=True)
                    selfies_list = file_data["selfies"]
                    assay_vals_list = file_data["assay_vals"]
                    file_samples_count = 0

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
                            
                            # Index this sample by its properties and property-value pairs
                            self._index_sample(sample_idx, reshaped)
                            file_samples_count += 1
                    
                    total_samples += file_samples_count
                    
                    # Update progress bar with detailed stats
                    pbar.set_postfix({
                        'samples': total_samples,
                        'file_samples': file_samples_count,
                        'failed_files': failed_files
                    })
                        
                except Exception as e:
                    failed_files += 1
                    logging.error(f"Error processing {file_path}: {e}")
                    pbar.set_postfix({
                        'samples': total_samples,
                        'failed_files': failed_files
                    })

    def _index_sample(self, sample_idx: int, property_value_pairs: Tensor):
        """Index a sample by its properties and property-value pairs."""
        properties = property_value_pairs[:, 0]  # Property tokens
        values = property_value_pairs[:, 1]      # Value tokens
        
        for prop_token, value_token in zip(properties, values):
            # Convert property token to property ID
            prop_id = (prop_token - self.tokenizer.selfies_offset).item()
            
            # Only index target properties
            if prop_id in self.target_properties:
                # Index by property
                self.property_sample_index[prop_id].append(sample_idx)
                
                # Index by property-value pair
                prop_val_key = (prop_id, value_token.item())
                self.property_value_index[prop_val_key].append(sample_idx)

    def _init_sampling_counters(self):
        """Initialize counters to track how often each property-value pair has been sampled."""
        self.property_value_counts = {}
        for prop_val_key in self.property_value_index.keys():
            self.property_value_counts[prop_val_key] = 0
    
    def _reset_epoch(self):
        """Reset epoch tracking to ensure full dataset coverage."""
        self.samples_seen_this_epoch = set()
        self.unseen_samples_queue = list(range(len(self.samples)))
        random.shuffle(self.unseen_samples_queue)
        logging.info(f"🔄 Starting epoch {self.current_epoch} with {len(self.unseen_samples_queue)} samples")

    def _select_sample_index(self) -> int:
        """
        Select a sample index using hybrid strategy:
        - Most of the time: balanced sampling to ensure property-value balance
        - Some of the time: coverage sampling to ensure all samples are seen
        """
        # If we've seen all samples this epoch, reset for next epoch
        if len(self.samples_seen_this_epoch) >= len(self.samples):
            self.current_epoch += 1
            self._reset_epoch()
        
        # Decide sampling strategy
        use_balanced_sampling = random.random() < self.balanced_ratio
        
        if use_balanced_sampling and self.property_value_counts:
            # Balanced sampling: select based on least sampled property-value pair
            target_prop_val = self._get_least_sampled_property_value()
            candidate_samples = self.property_value_index[target_prop_val]
            
            # Filter candidates to prioritize unseen samples if possible
            unseen_candidates = [s for s in candidate_samples if s not in self.samples_seen_this_epoch]
            
            if unseen_candidates:
                selected_sample_idx = random.choice(unseen_candidates)
            else:
                # All candidates have been seen, pick any
                selected_sample_idx = random.choice(candidate_samples)
        else:
            # Coverage sampling: ensure we see unseen samples
            if self.unseen_samples_queue:
                selected_sample_idx = self.unseen_samples_queue.pop()
            else:
                # Fallback to random if queue is empty (shouldn't happen)
                selected_sample_idx = random.randint(0, len(self.samples) - 1)
        
        # Track that we've seen this sample
        self.samples_seen_this_epoch.add(selected_sample_idx)
        
        return selected_sample_idx
            
    def _print_index_statistics(self):
        """Print statistics about the indexed properties."""
        logging.info("\n📈 Property indexing statistics:")
        for prop_id in sorted(self.target_properties):
            if prop_id in self.property_sample_index:
                sample_count = len(self.property_sample_index[prop_id])
                # Count unique values for this property
                prop_values = [key[1] for key in self.property_value_index.keys() if key[0] == prop_id]
                unique_values = len(set(prop_values))
                logging.info(f"   Property {prop_id}: {sample_count} samples, {unique_values} unique values")
            else:
                logging.info(f"   Property {prop_id}: 0 samples")

    def _get_least_sampled_property_value(self) -> Tuple[int, int]:
        """Get the property-value pair that has been sampled the least."""
        if not self.property_value_counts:
            # If no counts yet, return a random property-value pair
            return random.choice(list(self.property_value_index.keys()))
        
        # Find the minimum count
        min_count = min(self.property_value_counts.values())
        least_sampled = [pv for pv, count in self.property_value_counts.items() if count == min_count]
        return random.choice(least_sampled)

    def _update_sampling_counts(self, selected_properties: Tensor, selected_values: Tensor, mask: Tensor):
        """
        Update sampling counts for property-value pairs that were actually selected and seen by the model.
        
        Args:
            selected_properties: [nprops] tensor of selected property tokens (may include padding)
            selected_values: [nprops] tensor of selected value tokens (may include padding) 
            mask: [nprops] boolean tensor indicating which positions are valid (not padded)
        """
        # Only update counts for non-padded positions
        valid_properties = selected_properties[mask]
        valid_values = selected_values[mask]
        
        for prop_token, value_token in zip(valid_properties, valid_values):
            prop_id = (prop_token - self.tokenizer.selfies_offset).item()
            if prop_id in self.target_properties:
                prop_val_key = (prop_id, value_token.item())
                if prop_val_key in self.property_value_counts:
                    self.property_value_counts[prop_val_key] += 1

    def _process_sample_for_target_properties(self, property_value_pairs: Tensor, 
                                            least_sampled_prop_val: Tuple[int, int]) -> Tuple[Tensor, Tensor]:
        """
        Process sample to prioritize target properties and ensure they're included.
        The least sampled property-value pair is placed at a random position within the first nprops.
        
        Args:
            property_value_pairs: Tensor of shape [num_pairs, 2] with property-value pairs
            least_sampled_prop_val: The (prop_id, value_token) pair that was least sampled
        
        Returns:
            properties: [nprops] tensor of property tokens, padded if necessary
            values: [nprops] tensor of value tokens, padded if necessary
        """
        properties = property_value_pairs[:, 0]
        values = property_value_pairs[:, 1]
        
        # Find the least sampled property-value pair in this sample
        least_sampled_pair = None
        least_sampled_idx = None
        other_pairs = []
        
        for i, (prop_token, value_token) in enumerate(zip(properties, values)):
            prop_id = (prop_token - self.tokenizer.selfies_offset).item()
            pair = property_value_pairs[i:i+1]  # Keep as 2D tensor
            
            # Check if this is the least sampled property-value pair
            if (prop_id, value_token.item()) == least_sampled_prop_val:
                least_sampled_pair = pair
                least_sampled_idx = i
            else:
                other_pairs.append(pair)
        
        # Create output tensors
        properties_out = torch.full((self.nprops,), self.pad_idx, dtype=torch.long)
        values_out = torch.full((self.nprops,), self.pad_idx, dtype=torch.long)
        
        if least_sampled_pair is None:
            # This shouldn't happen if balanced sampling worked correctly,
            # but handle it gracefully by just using the first nprops pairs
            selected_pairs = other_pairs[:self.nprops]
        else:
            # Choose a random position for the least sampled pair within the first nprops
            if len(other_pairs) + 1 <= self.nprops:
                # We can fit all pairs, so choose any position
                random_position = random.randint(0, min(self.nprops - 1, len(other_pairs)))
            else:
                # We need to select other pairs, choose position within available slots
                random_position = random.randint(0, self.nprops - 1)
            
            # Build the final selection with least sampled pair at random position
            selected_pairs = []
            other_idx = 0
            
            for pos in range(self.nprops):
                if pos == random_position:
                    # Place the least sampled pair here
                    selected_pairs.append(least_sampled_pair)
                elif other_idx < len(other_pairs):
                    # Fill with other pairs
                    selected_pairs.append(other_pairs[other_idx])
                    other_idx += 1
                else:
                    # No more pairs to add
                    break
        
        # Convert selected pairs to output tensors
        for i, pair in enumerate(selected_pairs):
            if i >= self.nprops:
                break
            properties_out[i] = pair[0, 0]  # property token
            values_out[i] = pair[0, 1]     # value token
        
        return properties_out, values_out

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Get item using hybrid sampling strategy (ignores the provided idx).
        
        This method combines balanced sampling (for property-value balance) with 
        coverage sampling (to ensure all samples are eventually seen).
        """
        # Select sample using hybrid strategy
        selected_sample_idx = self._select_sample_index()
        selfies, property_value_pairs = self.samples[selected_sample_idx]
        
        # Get the current least sampled property-value pair for positioning
        target_prop_val = self._get_least_sampled_property_value()
        
        # Process the sample to prioritize the least sampled property at a random position
        properties, values = self._process_sample_for_target_properties(
            property_value_pairs, target_prop_val
        )
        
        # Apply tokenizer normalization
        mask = properties != self.pad_idx
        normproperties = self.tokenizer.norm_properties(properties, mask)
        normvalues = self.tokenizer.norm_values(values, mask)
        
        # Update counters ONLY for property-value pairs that were actually selected
        # and will be seen by the model (after truncation to nprops)
        self._update_sampling_counts(properties, values, mask)
        
        return selfies, normproperties, normvalues, mask

    def get_sampling_statistics(self) -> Dict:
        """Get statistics about how often each property-value pair has been sampled."""
        stats = {}
        for prop_id in self.target_properties:
            prop_stats = {}
            for (pid, value_token), count in self.property_value_counts.items():
                if pid == prop_id:
                    prop_stats[value_token] = count
            stats[prop_id] = prop_stats
        return stats

    def get_epoch_statistics(self) -> Dict:
        """Get statistics about current epoch coverage."""
        return {
            'current_epoch': self.current_epoch,
            'samples_seen_this_epoch': len(self.samples_seen_this_epoch),
            'total_samples': len(self.samples),
            'coverage_percent': (len(self.samples_seen_this_epoch) / len(self.samples)) * 100,
            'unseen_remaining': len(self.unseen_samples_queue),
            'balanced_ratio': self.balanced_ratio
        }

    def reset_sampling_counters(self):
        """Reset sampling counters to start fresh balancing."""
        self._init_sampling_counters()

    def force_new_epoch(self):
        """Force start of a new epoch (reset coverage tracking)."""
        self.current_epoch += 1
        self._reset_epoch()