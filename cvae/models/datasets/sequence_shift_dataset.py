import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch, torch.nn as nn, torch.nn.functional as F
import torch.utils.data
import json
from torch import Tensor
from typing import Tuple, Optional
from torch.utils.data import Dataset
from rotary_embedding_torch import RotaryEmbedding
from x_transformers import Encoder, Decoder
import math
import pathlib
import tqdm
import bisect
import random

from cvae.tokenizer.selfies_property_val_tokenizer import SelfiesPropertyValTokenizer
import cvae.utils

@torch.jit.script
def process_assay_vals(
    raw_assay_vals: Tensor,
    pad_idx: int,
    sep_idx: int,
    end_idx: int,
    nprops: int,
    assay_filter_tensor: Optional[Tensor] = None
) -> Tuple[Tensor, Tensor]:
    device = raw_assay_vals.device
    mask = raw_assay_vals != pad_idx
    valid_tokens = raw_assay_vals[mask][1:-1]

    if assay_filter_tensor is not None and assay_filter_tensor.numel() > 0:
        assay_ids = valid_tokens[::2]
        assay_mask = torch.isin(assay_ids, assay_filter_tensor)
        expanded_mask = torch.zeros_like(valid_tokens, dtype=torch.bool)
        expanded_mask[::2] = assay_mask
        expanded_mask[1::2] = assay_mask
        valid_tokens = valid_tokens[expanded_mask]

    reshaped = valid_tokens.view(-1, 2).contiguous()
    assert reshaped.numel() > 0, "No assay values found."

    perm = torch.randperm(reshaped.size(0))
    shuffled = reshaped[perm].flatten()
    av_truncate = shuffled[:nprops * 2]

    av_sos_eos = torch.cat([
        torch.tensor([sep_idx], device=device),
        av_truncate,
        torch.tensor([end_idx], device=device)
    ])
    out = F.pad(av_sos_eos, (0, nprops * 2 + 2 - av_sos_eos.size(0)), value=float(pad_idx))
    tch = torch.cat([torch.tensor([1], device=device), out[:-1]])
    return tch, out


class SequenceShiftDataset(Dataset):
    
    def __init__(self, path, tokenizer: SelfiesPropertyValTokenizer, nprops=5, assay_filter=[]):
        self.nprops = nprops
        self.assay_filter = assay_filter
        self.assay_filter_tensor = torch.tensor(assay_filter, dtype=torch.long) if len(assay_filter) > 0 else None
        self.file_paths = []
        self.file_lengths = []
        self.cumulative_lengths = [0]
        cumulative_length = 0
        self.tokenizer = tokenizer
        self.pad_idx, self.sep_idx, self.end_idx = tokenizer.PAD_IDX, tokenizer.SEP_IDX, tokenizer.END_IDX

        for file_path in tqdm.tqdm(pathlib.Path(path).glob("*.pt")):
            file_data = torch.load(file_path, map_location="cpu")
            selfies = file_data["selfies"]
            assay_vals = file_data["assay_vals"]
            
            valid_count = 0
            for s, a in zip(selfies, assay_vals):
                mask = a != self.pad_idx
                valid_tokens = a[mask][1:-1]
                if self.assay_filter_tensor is not None and self.assay_filter_tensor.numel() > 0:
                    assay_ids = valid_tokens[::2]
                    assay_mask = torch.isin(assay_ids, self.assay_filter_tensor)
                    if assay_mask.sum() == 0:
                        continue
                if valid_tokens.numel() >= 2:
                    valid_count += 1
            
            if valid_count > 0:
                self.file_paths.append(file_path)
                self.file_lengths.append(valid_count)
                cumulative_length += valid_count
                self.cumulative_lengths.append(cumulative_length)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0

    def __getitem__(self, idx):
        # Find which file contains this index
        file_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        local_idx = idx - self.cumulative_lengths[file_idx]
        
        # Load the file on-demand
        file_data = torch.load(self.file_paths[file_idx], map_location="cpu")
        selfies = file_data["selfies"]
        assay_vals = file_data["assay_vals"]
        
        # Find the local_idx-th valid sample in this file
        current_valid_idx = 0
        for s, a in zip(selfies, assay_vals):
            mask = a != self.pad_idx
            valid_tokens = a[mask][1:-1]
            if self.assay_filter_tensor is not None and self.assay_filter_tensor.numel() > 0:
                assay_ids = valid_tokens[::2]
                assay_mask = torch.isin(assay_ids, self.assay_filter_tensor)
                if assay_mask.sum() == 0:
                    continue
            if valid_tokens.numel() >= 2:
                if current_valid_idx == local_idx:
                    # Found our sample
                    tch, out = process_assay_vals(
                        a,
                        self.pad_idx,
                        self.sep_idx,
                        self.end_idx,
                        self.nprops,
                        self.assay_filter_tensor
                    )
                    return s, tch, out
                current_valid_idx += 1
        
        raise IndexError(f"Index {idx} not found")

