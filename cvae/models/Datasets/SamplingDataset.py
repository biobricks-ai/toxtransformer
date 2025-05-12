from torch.utils.data import Dataset
from collections import deque
import torch
import pathlib, bisect, random, tqdm
import torch.nn.functional as F
import pickle


class SamplingDataset(Dataset):
    """
    A PyTorch Dataset that dynamically samples compounds for training,
    favoring examples and properties that have not been seen recently.

    Key Features:
    - Tracks recency of sampled properties and dataset indices.
    - Penalizes candidates in the sampling pool based on recency rank.
    - Encourages uniform training coverage of both rare properties and datapoints.

    Args:
        path (str or Path): Directory with .pt files containing 'selfies' and 'assay_vals'.
        tokenizer: Tokenizer with PAD_IDX, SEP_IDX, END_IDX.
        nprops (int): Number of property-value pairs per sample.
        sample_pool (int): Number of candidates to score on each sample.
        recent_prop_cap (int): Capacity of the recent property queue.
        recent_idx_cap (int): Capacity of the recent index queue.
    """

    def __init__(self, path, tokenizer, nprops=5, sample_pool=512, recent_prop_cap=1000, recent_idx_cap=10000):
        self.nprops = nprops
        self.sample_pool = int(sample_pool)
        self.pad_idx, self.sep_idx, self.end_idx = tokenizer.PAD_IDX, tokenizer.SEP_IDX, tokenizer.END_IDX

        self.recent_props = deque(maxlen=recent_prop_cap)
        self.recent_prop_set = set()
        self.recency_rank = {}  # updated whenever recent_props changes

        self.recent_idxs = deque(maxlen=recent_idx_cap)
        self.idx_recency_rank = {}  # updated whenever recent_idxs changes

        self.propval_data = []  # List of List[(prop_id, val_id)]
        self.data = []
        self.cumulative_lengths = [0]
        self.path = str(path)

        total = 0
        files = list(pathlib.Path(path).glob("*.pt"))
        for file_path in tqdm.tqdm(files, total=len(files)):
            obj = torch.load(file_path, map_location="cpu")
            selfies, assay_vals = obj["selfies"], obj["assay_vals"]
            self.data.append((selfies, assay_vals))

            for row in assay_vals:
                mask = row != self.pad_idx
                items = row[mask][1:-1].view(-1, 2).tolist()
                self.propval_data.append([(int(p), int(v)) for p, v in items])

            total += selfies.size(0)
            self.cumulative_lengths.append(total)

    def __len__(self):
        return self.cumulative_lengths[-1]

    def _update_recent_props(self, selected_props):
        for p in selected_props:
            if p in self.recent_prop_set:
                self.recent_props.remove(p)
            self.recent_props.append(p)
        self.recent_prop_set = set(self.recent_props)

        if len(self.recent_props) == self.recent_props.maxlen:
            denom = len(self.recent_props) - 1
            self.recency_rank = {pid: i / denom for i, pid in enumerate(reversed(self.recent_props))}
        else:
            self.recency_rank = {pid: 1.0 for pid in self.recent_props}

    def _update_recent_idxs(self, idx):
        if idx in self.idx_recency_rank:
            self.recent_idxs.remove(idx)
        self.recent_idxs.append(idx)

        if len(self.recent_idxs) == self.recent_idxs.maxlen:
            denom = len(self.recent_idxs) - 1
            self.idx_recency_rank = {ix: i / denom for i, ix in enumerate(reversed(self.recent_idxs))}
        else:
            self.idx_recency_rank = {ix: 1.0 for ix in self.recent_idxs}

    def _sample_index(self):
        candidates = random.sample(range(len(self)), min(self.sample_pool, len(self)))
        if len(self.recent_props) < self.recent_props.maxlen:
            return random.choice(candidates)

        best_score = float("inf")
        best_idx = candidates[0]

        for idx in candidates:
            props = [p for p, _ in self.propval_data[idx]]
            prop_score = sum(self.recency_rank.get(p, 0.0) for p in props)
            idx_penalty = self.idx_recency_rank.get(idx, 0.0)
            score = prop_score + idx_penalty + random.uniform(0, 0.1)
            if score < best_score:
                best_score = score
                best_idx = idx

        return best_idx

    def __getitem__(self, _):
        idx = self._sample_index()
        self._update_recent_idxs(idx)

        file_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        local_idx = idx - self.cumulative_lengths[file_idx]
        selfies = self.data[file_idx][0][local_idx]
        pairs = self.propval_data[idx]

        # Choose nprops, preferring those less recently seen
        if len(self.recent_props) < self.recent_props.maxlen:
            selected = random.sample(pairs, min(len(pairs), self.nprops))
        else:
            scored = sorted(pairs, key=lambda x: self.recency_rank.get(x[0], 0.0))
            selected = scored[:self.nprops]

        random.shuffle(selected)
        self._update_recent_props([p for p, _ in selected])

        flat = [i for pair in selected for i in pair]
        device = selfies.device
        out = torch.tensor([self.sep_idx] + flat + [self.end_idx], device=device)
        out = F.pad(out, (0, self.nprops * 2 + 2 - out.size(0)), value=self.pad_idx)
        tch = torch.cat([torch.tensor([1], device=device), out[:-1]])

        return selfies, tch, out
