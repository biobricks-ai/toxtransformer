import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
from typing import Tuple, List, Dict
import pathlib, bisect, tqdm, random
from collections import defaultdict, Counter

@torch.jit.script
def process_assay_vals(raw_assay_vals: torch.Tensor, pad_idx: int, sep_idx: int, end_idx: int, nprops: int) -> Tuple[torch.Tensor, torch.Tensor]:
    mask = raw_assay_vals != pad_idx
    assay_vals = raw_assay_vals[mask][1:-1]
    reshaped = assay_vals.view(-1, 2).contiguous()

    assert reshaped.numel() > 0, "No assay values found."

    perm = torch.randperm(reshaped.size(0))
    shuffled = reshaped[perm].flatten()
    av_truncate = shuffled[: nprops * 2]

    device = raw_assay_vals.device
    av_sos_eos = torch.cat([
        torch.tensor([sep_idx], device=device),
        av_truncate,
        torch.tensor([end_idx], device=device)
    ])
    pad_value = float(pad_idx)
    out = F.pad(av_sos_eos, (0, nprops * 2 + 2 - av_sos_eos.size(0)), value=pad_value)
    tch = torch.cat([torch.tensor([1]), out[:-1]])
    
    return tch, out


class SamplingDataset(Dataset):
    def __init__(self, path, tokenizer, nprops=5, assay_filter: List[int] = [], min_freq=100):
        self.nprops = nprops
        self.tokenizer = tokenizer
        self.pad_idx, self.sep_idx, self.end_idx = tokenizer.PAD_IDX, tokenizer.SEP_IDX, tokenizer.END_IDX

        self.samples = []  # list of (selfies, assay_vals)
        self.assay_to_indices: Dict[int, List[int]] = defaultdict(list)
        self.assay_counts = Counter()

        # Load data
        for file_path in tqdm.tqdm(pathlib.Path(path).glob("*.pt")):
            file_data = torch.load(file_path, map_location="cpu")
            for i in range(file_data["selfies"].size(0)):
                selfies = file_data["selfies"][i]
                assay_vals = file_data["assay_vals"][i]
                idx = len(self.samples)

                # Skip empty values
                if (assay_vals != self.pad_idx).sum().item() <= 2:
                    continue

                self.samples.append((selfies, assay_vals))

                # Track assays
                assay_ids = assay_vals[1:-1:2]  # every second item (i.e., assay ID)
                for aid in assay_ids.tolist():
                    if not assay_filter or aid in assay_filter:
                        self.assay_to_indices[aid].append(idx)
                        self.assay_counts[aid] += 1

        # Create sampling weights for each assay (inverse freq or uniform if desired)
        self.min_freq = min_freq
        self.assay_sampling_weights = self._compute_sampling_weights()

    def _compute_sampling_weights(self):
        total = sum(self.assay_counts.values())
        weights = {
            aid: max(1.0, self.min_freq / count)
            for aid, count in self.assay_counts.items()
        }
        return weights

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Choose an assay to emphasize
        assay_ids = list(self.assay_sampling_weights.keys())
        assay_probs = torch.tensor([self.assay_sampling_weights[aid] for aid in assay_ids])
        assay_probs = assay_probs / assay_probs.sum()

        selected_assay = random.choices(assay_ids, weights=assay_probs.tolist(), k=1)[0]
        candidate_idxs = self.assay_to_indices[selected_assay]
        sampled_idx = random.choice(candidate_idxs)

        selfies_raw, raw_assay_vals = self.samples[sampled_idx]
        tch, out = process_assay_vals(raw_assay_vals, self.pad_idx, self.sep_idx, self.end_idx, self.nprops)
        return selfies_raw, tch, out
