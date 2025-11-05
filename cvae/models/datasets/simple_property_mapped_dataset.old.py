from typing import Dict, List, Tuple, Optional
import random
import torch
from torch.utils.data import Dataset
import pathlib
import tqdm
import logging
from torch import Tensor
from collections import defaultdict


class SimplePropertyMappedDataset(Dataset):
    """
    A simplified dataset that maps normalized property+value pairs to samples.
    Pre-normalizes all data during loading to avoid runtime normalization bugs.
    Tracks property sampling to preferentially select undersampled properties.
    
    The dataset:
    1. Pre-normalizes all property/value tokens during loading
    2. Creates a map from each normalized (property_id, value_id) pair to sample indices
    3. Stores all data in big arrays with consistent indexing
    4. Uses deterministic selection for reproducible results with optional seeding
    5. Tracks property sampling counts and preferentially samples underrepresented properties
    """

    def __init__(self, paths, tokenizer, target_properties: List[int], nprops: int = 1, 
                 seed: int = None, sampling_temp: float = 2.0):
        """
        Args:
            paths: Path(s) to data files
            tokenizer: The tokenizer to use
            target_properties: List of target properties to consider (pre-normalized IDs)
            nprops: Number of properties to return per sample
            seed: Random seed for deterministic shuffling of property-value pairs and indices
            sampling_temp: Temperature for weighted sampling (higher = more uniform, lower = more biased towards undersampled)
        """
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.PAD_IDX
        self.target_properties = target_properties
        self.nprops = nprops
        self.seed = seed
        self.sampling_temp = sampling_temp
        
        # Pre-normalized data storage
        self.all_selfies = []  # List of SELFIES tensors
        self.all_norm_properties = []  # List of normalized property tensors
        self.all_norm_values = []  # List of normalized value tensors
        
        # Map (norm_property_id, norm_value_id) -> list of sample indices
        self.property_value_to_indices: Dict[Tuple[int, int], List[int]] = {}
        
        # Keep track of all unique normalized property+value pairs
        self.property_value_pairs: List[Tuple[int, int]] = []
        
        # Track sampling counts for each property
        self.property_sample_counts: Dict[int, int] = defaultdict(int)
        self.total_property_samples = 0
        
        # Track which properties exist in each sample for efficient selection
        self.sample_properties: List[set] = []  # List of sets of normalized property IDs per sample
        
        # Create random number generator for consistent sampling
        self.rng = random.Random(seed) if seed is not None else random.Random()
        
        # Load and pre-process all samples
        self._load_and_preprocess_samples(paths)
        
        # Apply seeded shuffling if seed is provided
        if self.seed is not None:
            self._apply_seeded_shuffling()
        
        # Calculate dataset size
        self._calculate_dataset_size()
        
    def _load_and_preprocess_samples(self, paths):
        """Load samples and pre-normalize all property/value data."""
        
        file_paths = [pathlib.Path(p) for p in paths]
        logging.info(f"Loading and pre-processing samples for property+value pairs")
        
        sample_idx = 0
        
        for file_path in tqdm.tqdm(file_paths, desc="Loading files"):
            file_data = torch.load(file_path, map_location="cpu", weights_only=True, mmap=True)
            selfies_list = file_data["selfies"]
            assay_vals_list = file_data["assay_vals"]

            for selfies, assay_vals in zip(selfies_list, assay_vals_list):
                # Extract valid property-value pairs
                mask = assay_vals != self.pad_idx
                valid_tokens = assay_vals[mask][1:-1]  # Remove SEP and END tokens

                if len(valid_tokens) == 0 or len(valid_tokens) % 2 != 0:
                    continue  # Skip invalid samples

                property_value_pairs = valid_tokens.view(-1, 2).contiguous()  # [num_pairs, 2]
                
                # Pre-normalize ALL properties and values
                raw_properties = property_value_pairs[:, 0]
                raw_values = property_value_pairs[:, 1]
                
                # Apply normalization (same as tokenizer.norm_properties/norm_values)
                norm_properties = raw_properties - self.tokenizer.selfies_offset
                norm_values = raw_values - self.tokenizer.properties_offset
                
                # Store the sample data
                self.all_selfies.append(selfies)
                self.all_norm_properties.append(norm_properties)
                self.all_norm_values.append(norm_values)
                
                # Track which properties are in this sample
                sample_prop_set = set(norm_properties.tolist())
                self.sample_properties.append(sample_prop_set)
                
                # Index by normalized property+value pairs
                for norm_prop, norm_val in zip(norm_properties, norm_values):
                    norm_prop_id = norm_prop.item()
                    norm_val_id = norm_val.item()
                    
                    # Only consider target properties (using normalized IDs)
                    if norm_prop_id in self.target_properties:
                        key = (norm_prop_id, norm_val_id)
                        
                        # Initialize if first time seeing this normalized pair
                        if key not in self.property_value_to_indices:
                            self.property_value_to_indices[key] = []
                            self.property_value_pairs.append(key)
                        
                        self.property_value_to_indices[key].append(sample_idx)
                
                sample_idx += 1

        # Sort property_value_pairs for consistent ordering before shuffling
        self.property_value_pairs.sort()
        
        # Initialize property counts based on available data
        self._initialize_property_counts()
        
        # Print statistics
        self._print_statistics()
    
    def _initialize_property_counts(self):
        """Initialize property sampling counts based on data distribution."""
        # Count how many samples each property appears in
        property_appearance_counts = defaultdict(int)
        
        for sample_props in self.sample_properties:
            for prop_id in sample_props:
                property_appearance_counts[prop_id] += 1
        
        # Initialize sampling counts inversely proportional to appearance
        # This gives underrepresented properties a head start
        if property_appearance_counts:
            max_count = max(property_appearance_counts.values())
            for prop_id, count in property_appearance_counts.items():
                # Give inverse weight as initial count
                initial_count = max(1, max_count - count)
                self.property_sample_counts[prop_id] = initial_count
                self.total_property_samples += initial_count
    
    def _apply_seeded_shuffling(self):
        """Apply seeded shuffling to property-value pairs and their indices."""
        logging.info(f"Applying seeded shuffling with seed: {self.seed}")
        
        # Shuffle the order of property-value pairs
        self.rng.shuffle(self.property_value_pairs)
        
        # Shuffle the indices within each property-value pair
        for key in self.property_value_to_indices:
            self.rng.shuffle(self.property_value_to_indices[key])
        
        logging.info("Seeded shuffling completed")
    
    def _print_statistics(self):
        """Print statistics about normalized property+value pair mappings."""
        logging.info(f"\nProperty+Value mapping statistics:")
        logging.info(f"Total samples loaded: {len(self.all_selfies)}")
        
        # Group by property for cleaner output
        property_stats = {}
        for (norm_prop_id, norm_val_id), indices in self.property_value_to_indices.items():
            if norm_prop_id not in property_stats:
                property_stats[norm_prop_id] = []
            property_stats[norm_prop_id].append((norm_val_id, len(indices)))
        
        for norm_prop_id in sorted(property_stats.keys()):
            value_stats = sorted(property_stats[norm_prop_id])
            total_samples = sum(count for _, count in value_stats)
            unique_values = len(value_stats)
            logging.info(f"   Property {norm_prop_id}: {unique_values} unique values, {total_samples} total samples")
            
            # Show top 5 most frequent values for this property
            value_stats.sort(key=lambda x: x[1], reverse=True)
            for norm_val_id, count in value_stats[:5]:
                logging.info(f"     Value {norm_val_id}: {count} samples")
            if len(value_stats) > 5:
                logging.info(f"     ... and {len(value_stats) - 5} more values")
        
        logging.info(f"   Total unique normalized property+value pairs: {len(self.property_value_pairs)}")
        if self.seed is not None:
            logging.info(f"   Shuffled with seed: {self.seed}")
        
        # Print initial sampling weights
        logging.info(f"\nInitial property sampling weights:")
        sorted_props = sorted(self.property_sample_counts.items(), key=lambda x: x[1])
        for prop_id, count in sorted_props[:10]:
            weight = count / max(1, self.total_property_samples)
            logging.info(f"   Property {prop_id}: initial weight = {weight:.4f}")
    
    def _calculate_dataset_size(self):
        """Calculate total dataset size based on property+value pair mappings."""
        self.total_samples = sum(len(indices) for indices in self.property_value_to_indices.values())
        logging.info(f"Total dataset size: {self.total_samples} samples")
    
    def _get_sampling_weights(self, available_properties: List[int]) -> List[float]:
        """
        Calculate sampling weights for available properties based on how undersampled they are.
        
        Args:
            available_properties: List of property IDs available for sampling
            
        Returns:
            List of weights corresponding to each property
        """
        if not available_properties:
            return []
        
        # Calculate inverse frequency weights
        weights = []
        for prop_id in available_properties:
            # Add 1 to avoid division by zero
            sample_count = self.property_sample_counts.get(prop_id, 0) + 1
            # Inverse frequency weight
            weight = 1.0 / sample_count
            weights.append(weight)
        
        # Apply temperature to control sampling distribution
        if self.sampling_temp != 1.0 and self.sampling_temp > 0:
            weights = [w ** (1.0 / self.sampling_temp) for w in weights]
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            # Uniform if all weights are zero
            weights = [1.0 / len(weights)] * len(weights)
        
        return weights
    
    def _weighted_sample(self, population: List, weights: List[float], k: int) -> List:
        """
        Sample k items from population according to weights without replacement.
        
        Args:
            population: List of items to sample from
            weights: Sampling weights for each item
            k: Number of items to sample
            
        Returns:
            List of sampled items
        """
        if not population or k <= 0:
            return []
        
        k = min(k, len(population))
        
        # Use weighted sampling without replacement
        sampled = []
        remaining_pop = list(population)
        remaining_weights = list(weights)
        
        for _ in range(k):
            if not remaining_pop:
                break
            
            # Normalize remaining weights
            total = sum(remaining_weights)
            if total > 0:
                probs = [w / total for w in remaining_weights]
            else:
                probs = [1.0 / len(remaining_weights)] * len(remaining_weights)
            
            # Sample one item
            idx = self.rng.choices(range(len(remaining_pop)), weights=probs)[0]
            sampled.append(remaining_pop[idx])
            
            # Remove sampled item
            remaining_pop.pop(idx)
            remaining_weights.pop(idx)
        
        return sampled

    def _select_properties(self, norm_properties: Tensor, norm_values: Tensor, 
                          target_norm_prop_id: int, target_norm_val_id: int) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Select nprops properties, ensuring target property+value is included at a random position.
        Uses weighted sampling to prefer undersampled properties for off-target slots.
        """
        # Find target property+value pair and other pairs
        target_indices = []
        other_property_indices = defaultdict(list)  # Group by property ID
        
        for i, (norm_prop, norm_val) in enumerate(zip(norm_properties, norm_values)):
            norm_prop_id = norm_prop.item()
            norm_val_id = norm_val.item()
            
            if norm_prop_id == target_norm_prop_id and norm_val_id == target_norm_val_id:
                target_indices.append(i)
            else:
                other_property_indices[norm_prop_id].append(i)
        
        # Select indices for output
        selected_indices = []
        selected_prop_ids = []
        
        # Guarantee target property+value is included
        if target_indices:
            target_idx = self.rng.choice(target_indices)
            selected_prop_ids.append(target_norm_prop_id)
        else:
            # This should not happen if the indexing is correct, but handle gracefully
            logging.warning(f"Target property+value pair ({target_norm_prop_id}, {target_norm_val_id}) not found in sample")
            target_idx = None
        
        # Select other properties using weighted sampling
        num_others_needed = self.nprops - 1 if target_idx is not None else self.nprops
        
        if other_property_indices and num_others_needed > 0:
            # Get list of available properties (excluding target)
            available_props = list(other_property_indices.keys())
            
            # Calculate sampling weights based on how undersampled each property is
            weights = self._get_sampling_weights(available_props)
            
            # Sample properties according to weights
            selected_props = self._weighted_sample(available_props, weights, num_others_needed)
            
            # For each selected property, randomly choose one of its indices
            selected_other_indices = []
            for prop_id in selected_props:
                prop_indices = other_property_indices[prop_id]
                selected_idx = self.rng.choice(prop_indices)
                selected_other_indices.append(selected_idx)
                selected_prop_ids.append(prop_id)
        else:
            selected_other_indices = []
        
        # Update sampling counts for selected properties
        for prop_id in selected_prop_ids:
            self.property_sample_counts[prop_id] += 1
            self.total_property_samples += 1
        
        # Combine target and other indices
        if target_idx is not None:
            all_selected = [target_idx] + selected_other_indices
        else:
            all_selected = selected_other_indices
        
        # Shuffle the combined list to randomize the position of the target
        self.rng.shuffle(all_selected)
        
        # Take only nprops indices
        selected_indices = all_selected[:self.nprops]
        num_real_props = len(selected_indices)
        
        # Convert to tensors and select
        if selected_indices:
            selected_indices = torch.tensor(selected_indices)
            selected_properties = norm_properties[selected_indices]
            selected_values = norm_values[selected_indices]
        else:
            # Fallback - should not happen
            selected_properties = norm_properties[:self.nprops]
            selected_values = norm_values[:self.nprops]
            num_real_props = min(len(norm_properties), self.nprops)
        
        # Pad if necessary and create mask based on real vs padded
        if len(selected_properties) < self.nprops:
            pad_size = self.nprops - len(selected_properties)
            # Use 0 as padding value - this should be valid for normalized embeddings
            pad_properties = torch.zeros(pad_size, dtype=selected_properties.dtype)
            pad_values = torch.zeros(pad_size, dtype=selected_values.dtype)
            selected_properties = torch.cat([selected_properties, pad_properties])
            selected_values = torch.cat([selected_values, pad_values])
        
        # Create mask: True for real properties, False for padded
        mask = torch.zeros(self.nprops, dtype=torch.bool)
        mask[:num_real_props] = True
        
        return selected_properties, selected_values, mask

    def __len__(self):
        # Maximum samples any single property-value pair has
        if not self.property_value_to_indices:
            return 0
        max_samples_per_pair = max(len(indices) for indices in self.property_value_to_indices.values())
        # Total combinations: each pair × max samples per pair
        return len(self.property_value_pairs) * max_samples_per_pair

    def __getitem__(self, idx):
        """
        Get item using deterministic property+value pair selection.
        Now works with pre-normalized data and tracks property sampling.
        """
        # Determine which property+value pair and which sample
        num_property_value_pairs = len(self.property_value_pairs)
        pair_idx = idx % num_property_value_pairs
        sample_idx_for_pair = idx // num_property_value_pairs
        
        # Get the target normalized property+value pair
        target_norm_prop_id, target_norm_val_id = self.property_value_pairs[pair_idx]
        
        # Get sample indices for this property+value pair
        indices_for_pair = self.property_value_to_indices[(target_norm_prop_id, target_norm_val_id)]
        
        # Select sample (with wraparound)
        actual_sample_idx = sample_idx_for_pair % len(indices_for_pair)
        sample_idx = indices_for_pair[actual_sample_idx]
        
        # Get the pre-stored, pre-normalized data
        selfies = self.all_selfies[sample_idx]
        norm_properties = self.all_norm_properties[sample_idx]
        norm_values = self.all_norm_values[sample_idx]
        
        # Select properties with target property+value guaranteed to be included
        # This now uses weighted sampling for off-target properties
        selected_properties, selected_values, mask = self._select_properties(
            norm_properties, norm_values, target_norm_prop_id, target_norm_val_id
        )
        
        return selfies, selected_properties, selected_values, mask

    def get_property_value_sample_count(self, norm_property_id: int, norm_value_id: int) -> int:
        """Get number of samples for a specific normalized property+value pair."""
        return len(self.property_value_to_indices.get((norm_property_id, norm_value_id), []))

    def get_all_property_value_counts(self) -> Dict[Tuple[int, int], int]:
        """Get sample counts for all normalized property+value pairs."""
        return {pair: len(indices) for pair, indices in self.property_value_to_indices.items()}
    
    def get_property_value_pairs(self) -> List[Tuple[int, int]]:
        """Get all unique normalized property+value pairs."""
        return self.property_value_pairs.copy()
    
    def get_values_for_property(self, norm_property_id: int) -> List[int]:
        """Get all unique normalized values for a specific normalized property."""
        values = []
        for norm_prop_id, norm_value_id in self.property_value_pairs:
            if norm_prop_id == norm_property_id:
                values.append(norm_value_id)
        return sorted(list(set(values)))
    
    def get_property_sampling_stats(self) -> Dict[int, Dict[str, float]]:
        """
        Get sampling statistics for each property.
        
        Returns:
            Dictionary mapping property ID to stats including sample count and sampling rate
        """
        stats = {}
        for prop_id, count in self.property_sample_counts.items():
            stats[prop_id] = {
                'sample_count': count,
                'sampling_rate': count / max(1, self.total_property_samples),
                'appearance_count': sum(1 for props in self.sample_properties if prop_id in props)
            }
        return stats
    
    def reset_sampling_counts(self):
        """Reset the sampling counts for all properties."""
        logging.info("Resetting property sampling counts")
        self.property_sample_counts.clear()
        self.total_property_samples = 0
        self._initialize_property_counts()