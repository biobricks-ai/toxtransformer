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

class InMemorySequenceShiftDataset(Dataset):
    
    def __init__(self, path, tokenizer, nprops=5, assay_filter: List[int] = []):
        self.nprops = nprops
        self.tokenizer = tokenizer
        self.pad_idx, self.sep_idx, self.end_idx = tokenizer.PAD_IDX, tokenizer.SEP_IDX, tokenizer.END_IDX
        self.assay_filter_tensor = torch.tensor(assay_filter, dtype=torch.long) if assay_filter else None

        # Each sample is (selfies_tensor, reshaped_valid_tokens [N x 2])
        self.samples: List[Tuple[Tensor, Tensor]] = []

        self.sep_idx = tokenizer.SEP_IDX
        self.end_idx = tokenizer.END_IDX
        self.one_token = 1

        for file_path in tqdm.tqdm(pathlib.Path(path).glob("*.pt"), desc="Loading dataset into RAM"):
            file_data = torch.load(file_path, map_location="cpu")
            selfies_list = file_data["selfies"]
            assay_vals_list = file_data["assay_vals"]

            for selfies, assay_vals in zip(selfies_list, assay_vals_list):
                mask = assay_vals != self.pad_idx
                valid_tokens = assay_vals[mask][1:-1]

                if self.assay_filter_tensor is not None and self.assay_filter_tensor.numel() > 0:
                    assay_ids = valid_tokens[::2]
                    assay_mask = torch.isin(assay_ids, self.assay_filter_tensor)
                    expanded_mask = torch.zeros_like(valid_tokens, dtype=torch.bool)
                    expanded_mask[::2] = assay_mask
                    expanded_mask[1::2] = assay_mask
                    valid_tokens = valid_tokens[expanded_mask]

                if valid_tokens.numel() >= 2:
                    reshaped = valid_tokens.view(-1, 2).contiguous()
                    self.samples.append((selfies, reshaped))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        selfies, reshaped = self.samples[idx]
        tch, out = self._process(reshaped)
        return selfies, tch, out

    def _process(self, reshaped: Tensor) -> Tuple[Tensor, Tensor]:
        # Work entirely with CPU tensors, let pin_memory handle GPU transfer
        perm = torch.randperm(reshaped.size(0))
        shuffled = reshaped[perm].flatten()
        av_truncate = shuffled[:self.nprops * 2]
        
        # Create tensors on CPU
        av_sos_eos = torch.cat([
            torch.tensor([self.sep_idx]), 
            av_truncate, 
            torch.tensor([self.end_idx])
        ])
        
        out = F.pad(av_sos_eos, (0, self.nprops * 2 + 2 - av_sos_eos.size(0)), value=self.pad_idx)
        tch = torch.cat([torch.tensor([self.one_token]), out[:-1]])
        return tch, out

class PreloadedSequenceShiftDataset:
    def __init__(self, path, tokenizer):
        self.samples: List[Tuple[Tensor, Tensor]] = []
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.PAD_IDX
        self.property_to_sample_idxs: Dict[int, List[int]] = defaultdict(list)
        self.property_to_value_to_sample_idxs: Dict[int, Dict[int, List[int]]] = defaultdict(lambda: defaultdict(list))
        self.sample_to_numprops : Dict[int, int] = defaultdict(int)

        sample_idx = 0
        for file_path in tqdm.tqdm(pathlib.Path(path).glob("*.pt"), desc="Preloading dataset"):
            file_data = torch.load(file_path, map_location="cpu")
            selfies_list = file_data["selfies"]
            assay_vals_list = file_data["assay_vals"]

            for selfies, assay_vals in zip(selfies_list, assay_vals_list):
                mask = assay_vals != self.pad_idx
                valid_tokens = assay_vals[mask][1:-1]
                reshaped = valid_tokens.view(-1, 2).contiguous()
                self.samples.append((selfies, reshaped))
                
                # For each property (assay id) in this sample, add sample_idx to the dict
                for prop in reshaped[:, 0].unique().tolist():
                    self.property_to_sample_idxs[prop].append(sample_idx)

                # Map property values to sample indices
                for prop, value in zip(reshaped[:, 0].tolist(), reshaped[:, 1].tolist()):
                    self.property_to_value_to_sample_idxs[prop][value].append(sample_idx)

                self.sample_to_numprops[sample_idx] = reshaped.size(0)

                sample_idx += 1

class TargetPropertySequenceShiftWrapper(torch.utils.data.Dataset):

    def __init__(self, base: PreloadedSequenceShiftDataset, target_property_token: int, nprops: int, nsamples: int, minprops: int):
        self.tokenizer = base.tokenizer
        self.nprops = nprops
        self.pad_idx = self.tokenizer.PAD_IDX
        self.target_property_token = target_property_token
        self.SEP = torch.tensor([self.tokenizer.SEP_IDX], dtype=torch.long)
        self.END = torch.tensor([self.tokenizer.END_IDX], dtype=torch.long)

        self.base_samples = base.samples
        
        # Get all unique values for the target property
        value_to_idxs = base.property_to_value_to_sample_idxs[target_property_token]
        
        # For each value, collect sample indices with at least nprops properties
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
            self.samples.extend(idxs[:per_value_samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_idx = self.samples[idx]
        selfies, reshaped = self.base_samples[base_idx]
        tch, out = self._process(reshaped)
        return selfies, tch, out, reshaped.size(0)

    def _process(self, reshaped: Tensor) -> Tuple[Tensor, Tensor]:
        device = reshaped.device

        # Split out the target row
        target_rows = reshaped[reshaped[:, 0] == self.target_property_token]
        other_rows = reshaped[reshaped[:, 0] != self.target_property_token]

        if target_rows.size(0) == 0:
            raise ValueError("Target property token not found in reshaped tensor.")

        # Shuffle other rows
        perm = torch.randperm(other_rows.size(0))
        shuffled = other_rows[perm]

        # Keep only the first (nprops - 1) from shuffled, append target last
        n_other = min(self.nprops - 1, shuffled.size(0))
        selected = torch.cat([shuffled[:n_other], target_rows[0:1]], dim=0).flatten()

        av_sos_eos = torch.cat([self.SEP.to(device), selected, self.END.to(device)])
        total_len = self.nprops * 2 + 2
        out = F.pad(av_sos_eos, (0, total_len - av_sos_eos.size(0)), value=float(self.pad_idx))
        tch = torch.cat([torch.tensor([1], device=device), out[:-1]])

        return tch, out
