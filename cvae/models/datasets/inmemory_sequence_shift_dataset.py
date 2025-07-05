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

        self.SEP = torch.tensor([tokenizer.SEP_IDX], dtype=torch.long)
        self.END = torch.tensor([tokenizer.END_IDX], dtype=torch.long)

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
        device = reshaped.device
        perm = torch.randperm(reshaped.size(0))
        shuffled = reshaped[perm].flatten()
        av_truncate = shuffled[:self.nprops * 2] # Flatten to 1D and truncate to nprops * 2 tokens
        av_sos_eos = torch.cat([self.SEP.to(device), av_truncate, self.END.to(device)])
        out = F.pad(av_sos_eos, (0, self.nprops * 2 + 2 - av_sos_eos.size(0)), value=float(self.pad_idx))
        tch = torch.cat([torch.tensor([1], device=device), out[:-1]])
        return tch, out

class PreloadedSequenceShiftDataset:
    def __init__(self, path, tokenizer):
        self.samples: List[Tuple[Tensor, Tensor]] = []
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.PAD_IDX

        for file_path in tqdm.tqdm(pathlib.Path(path).glob("*.pt"), desc="Preloading dataset"):
            file_data = torch.load(file_path, map_location="cpu")
            selfies_list = file_data["selfies"]
            assay_vals_list = file_data["assay_vals"]

            for selfies, assay_vals in zip(selfies_list, assay_vals_list):
                mask = assay_vals != self.pad_idx
                valid_tokens = assay_vals[mask][1:-1]
                if valid_tokens.numel() >= 2:
                    reshaped = valid_tokens.view(-1, 2).contiguous()
                    self.samples.append((selfies, reshaped))


class PreloadedSequenceShiftWrapper(torch.utils.data.Dataset):
    def __init__(self, base: PreloadedSequenceShiftDataset, nprops=5, assay_filter: List[int] = []):
        self.tokenizer = base.tokenizer
        self.nprops = nprops
        self.pad_idx = self.tokenizer.PAD_IDX
        self.SEP = torch.tensor([self.tokenizer.SEP_IDX], dtype=torch.long)
        self.END = torch.tensor([self.tokenizer.END_IDX], dtype=torch.long)

        self.samples = []
        self.base_samples = base.samples
        self.assay_filter_tensor = torch.tensor(assay_filter, dtype=torch.long) if assay_filter else None

        for idx, (selfies, reshaped) in enumerate(self.base_samples):
            if self.assay_filter_tensor is not None:
                assay_ids = reshaped[:, 0]
                if not torch.any(torch.isin(assay_ids, self.assay_filter_tensor)):
                    continue
            self.samples.append(idx)  # Just store index for indirection

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        base_idx = self.samples[idx]
        selfies, reshaped = self.base_samples[base_idx]
        return selfies, *self._process(reshaped)

    def _process(self, reshaped: Tensor) -> Tuple[Tensor, Tensor]:
        device = reshaped.device
        perm = torch.randperm(reshaped.size(0))
        shuffled = reshaped[perm].flatten()
        av_truncate = shuffled[:self.nprops * 2]
        av_sos_eos = torch.cat([self.SEP.to(device), av_truncate, self.END.to(device)])
        out = F.pad(av_sos_eos, (0, self.nprops * 2 + 2 - av_sos_eos.size(0)), value=float(self.pad_idx))
        tch = torch.cat([torch.tensor([1], device=device), out[:-1]])
        return tch, out

class TargetPropertySequenceShiftWrapper(torch.utils.data.Dataset):
    def __init__(self, base: PreloadedSequenceShiftDataset, target_property_token: int, nprops: int = 5):
        self.tokenizer = base.tokenizer
        self.nprops = nprops
        self.pad_idx = self.tokenizer.PAD_IDX
        self.target_property_token = target_property_token
        self.SEP = torch.tensor([self.tokenizer.SEP_IDX], dtype=torch.long)
        self.END = torch.tensor([self.tokenizer.END_IDX], dtype=torch.long)

        self.base_samples = base.samples
        self.samples = []

        for idx, (selfies, reshaped) in enumerate(self.base_samples):
            if torch.any(reshaped[:, 0] == target_property_token):
                self.samples.append(idx)

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
