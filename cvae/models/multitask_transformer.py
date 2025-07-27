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

class LearnedPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Learnable embeddings for the positions
        self.positional_embeddings = nn.Embedding(max_len, d_model)

    def forward(self, x):
        # Create a tensor of position indices [0, 1, 2, ..., sequence_length-1]
        position_indices = torch.arange(x.size(1), dtype=torch.long, device=x.device).unsqueeze(0).expand(x.size(0), x.size(1))
        
        # Retrieve the positional embeddings for the position indices
        position_embeddings = self.positional_embeddings(position_indices)
        
        # Add the positional embeddings to the input embeddings
        x = x + position_embeddings
        
        return self.dropout(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)


def generate_square_subsequent_mask(sz: int) -> torch.Tensor:
    """Generate a standard causal mask (upper triangular with -inf above diag)"""
    return torch.triu(torch.full((sz, sz), float('-inf')), diagonal=1)

@torch.no_grad()
def generate_custom_subsequent_mask(sz: int, device=None) -> torch.Tensor:
    mask = torch.full((sz, sz), float('-inf'), device=device)
    mask = torch.triu(mask, diagonal=1)
    for i in range(1, sz - 1, 2):
        mask[i, i + 1] = 0.0  # allow peeking
    return mask


class TokenEmbedding(nn.Module):

    def __init__(self, tokenizer, hdim=256):
        super().__init__()
        self.tokenizer = tokenizer
        self.hdim = hdim
        self.embedding = nn.Embedding(tokenizer.vocab_size, hdim)
        self.embedding_norm = nn.LayerNorm(hdim)

    def forward(self, input):
        embedded = self.embedding(input)
        return self.embedding_norm(embedded)

class MultitaskTransformer(nn.Module):

    def __init__(self, tokenizer, hdim=256, nhead=2, num_layers=4, 
        dim_feedforward=256, dropout_rate=0.1, output_size=None,
        shared_embedding=None):

        super().__init__()

        self.output_size = tokenizer.vocab_size if output_size is None else output_size
        self.hdim = hdim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.tokenizer = tokenizer
        self.token_pad_idx = tokenizer.PAD_IDX

        self.embedding = TokenEmbedding(tokenizer, hdim) if shared_embedding is None else shared_embedding
        
        self.encoder = Encoder(
            dim=self.hdim,
            depth=num_layers,
            heads=self.nhead,
            ff_mult=dim_feedforward // hdim,
            rotary_pos_emb=True,
            attn_dropout=dropout_rate,
            ff_dropout=dropout_rate
        )

        self.decoder = Decoder(
            dim=self.hdim,
            depth=num_layers,
            heads=self.nhead,
            ff_mult=dim_feedforward // hdim,
            rotary_pos_emb=True,
            attn_dropout=dropout_rate,
            ff_dropout=dropout_rate,
            cross_attend=True
        )

        self.decoder_norm = nn.LayerNorm(self.hdim)
        self.classification_layers = nn.Sequential(
            nn.LayerNorm(self.hdim),
            nn.Linear(self.hdim, self.hdim),
            nn.ReLU(),
            nn.Linear(self.hdim, self.output_size)
        )

    def forward(self, input, teach_forcing):
        memory_mask = input != self.token_pad_idx
        tgt_mask = teach_forcing != self.token_pad_idx

        input_embedding = self.embedding(input)
        input_encoding = self.encoder(input_embedding, mask=memory_mask)

        teach_forcing_emb = self.embedding(teach_forcing)

        decoded = self.decoder(
            teach_forcing_emb,
            context=input_encoding,
            mask=tgt_mask,
            context_mask=memory_mask
        )

        decoded = self.decoder_norm(decoded)
        logits = self.classification_layers(decoded)
        return logits

    # @staticmethod
    # def lossfn(ignore_index=-100, weight_decay=1e-5):
    #     ce_lossfn = nn.CrossEntropyLoss(reduction='mean', ignore_index=ignore_index, label_smoothing=0.01)

    #     def lossfn(parameters, logits, output):
    #         loss = ce_lossfn(logits, output)
    #         return loss

    #     return lossfn

    # @staticmethod
    # def build_eval_lossfn(value_indexes, device):
    #     value_token_ids = torch.tensor(list(value_indexes), dtype=torch.long).to(device)
    #     ce_lossfn = nn.CrossEntropyLoss(reduction='mean', ignore_index=-100, label_smoothing=0.01)

    #     def lossfn(parameters, logits, output):
    #         B, V, T = logits.shape
    #         logits = logits.permute(0, 2, 1).contiguous()
    #         logits_flat = logits.reshape(-1, V)
    #         output_flat = output.reshape(-1)

    #         mask = torch.isin(output_flat, value_token_ids)
    #         if mask.sum() == 0:
    #             return torch.tensor(0.0, device=output.device)

    #         logits_sel = logits_flat[mask]
    #         output_sel = output_flat[mask]
    #         return ce_lossfn(logits_sel, output_sel)

    #     return lossfn

    def save(self, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        cvae.utils.mk_empty_directory(path, overwrite=True)
        cvae.utils.mk_empty_directory(path / "spvt_tokenizer", overwrite=True)
        self.tokenizer.save(path / "spvt_tokenizer")
        torch.save(self.state_dict(), path / "mtransformer.pt")
        return path

    @staticmethod
    def load(dirpath = pathlib.Path("brick/mtransform1")):
        dirpath = pathlib.Path(dirpath)
        tokenizer = SelfiesPropertyValTokenizer.load(dirpath / "spvt_tokenizer")
        model = MultitaskTransformer(tokenizer)
        model.load_state_dict(torch.load(dirpath / 'mtransformer.pt'))
        model.eval()
        return model

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

class LabelSmoothingCrossEntropySequence(nn.Module):
    def __init__(self, epsilon_ls=0.05, ignore_index=None):
        super(LabelSmoothingCrossEntropySequence, self).__init__()
        self.epsilon_ls = epsilon_ls
        self.ignore_index = ignore_index

    def forward(self, out, tgt):
        num_classes = out.size(-1)
        dev = out.device
        
        if tgt.dim() == 1:
            # Masked case: (num_tokens,)
            fill = self.epsilon_ls / (num_classes - 1)
            with torch.no_grad():
                smooth_label = torch.full(size=(tgt.size(0), num_classes), fill_value=fill, device=dev)
                tgt = tgt.unsqueeze(-1)
                smooth_label.scatter_(-1, tgt, 1.0 - self.epsilon_ls)

                if self.ignore_index is not None:
                    ignore_mask = tgt.eq(self.ignore_index)
                    smooth_label.masked_fill_(ignore_mask, 0.0)

            loss = -smooth_label * F.log_softmax(out, dim=-1)

            if self.ignore_index is not None:
                loss.masked_fill_(ignore_mask, 0.0)

            loss = loss.sum(dim=-1)
            loss = loss.mean()

        else:
            # Regular case: (batch_size, seq_len)
            batch_size, seq_len = tgt.size()
            fill = self.epsilon_ls / (num_classes - 1)
            with torch.no_grad():
                smooth_label = torch.full(size=(batch_size, seq_len, num_classes), fill_value=fill, device=dev)
                tgt = tgt.unsqueeze(-1)
                smooth_label.scatter_(-1, tgt, 1.0 - self.epsilon_ls)

                if self.ignore_index is not None:
                    ignore_mask = tgt.eq(self.ignore_index)
                    smooth_label.masked_fill_(ignore_mask, 0.0)

            loss = -smooth_label * F.log_softmax(out, dim=-1)

            if self.ignore_index is not None:
                loss.masked_fill_(ignore_mask, 0.0)

            loss = loss.sum(dim=-1)
            loss = loss.masked_select(~ignore_mask.squeeze(-1)).mean()

        return loss


class NoamLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Learning rate scheduler from "Attention is All You Need" (Vaswani et al.).
    
    Arguments:
        optimizer: The optimizer to modify
        model_size: Hidden dimension size of the model (d_model in the paper)
        warmup_steps: Number of warmup steps before starting decay
        last_epoch: The index of the last epoch (-1 default)
        multiplier: Overall scale factor for the learning rate
        max_lr: Maximum learning rate (after applying multiplier)
        min_lr: Minimum learning rate to clip at
    """
    def __init__(self, optimizer, model_size, warmup_steps, last_epoch=-1, multiplier=1.0, max_lr=5e-5, min_lr=1e-6):
        self.model_size = model_size
        self.warmup_steps = warmup_steps
        self.multiplier = multiplier
        # Store the true maximum learning rate (after multiplier)
        self.true_max_lr = max_lr
        self.max_lr = max_lr
        self.min_lr = min_lr
        super(NoamLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        step_num = self.last_epoch + 1
        
        # Calculate the Noam scale factor
        scale = self.model_size ** (-0.5) * min(
            step_num ** (-0.5), 
            step_num * self.warmup_steps ** (-1.5)
        )
        
        # Apply multiplier to the scale
        scaled_lr = self.max_lr * self.multiplier * scale
        
        # Generate learning rates for all parameter groups
        lrs = [scaled_lr for _ in self.base_lrs]
        
        # Clip learning rates between min_lr and true_max_lr
        lrs = [min(lr, self.true_max_lr) for lr in lrs]
        lrs = [max(lr, self.min_lr) for lr in lrs]
        
        return lrs

from torch.optim.lr_scheduler import LambdaLR

def linear_warmup_and_decay_scheduler(optimizer, warmup_steps, total_steps, max_lr=1.0, min_lr=0.0):
    """
    GPT-3 style scheduler: linear warmup from min_lr to max_lr, then linear decay back to min_lr.
    Assumes max_lr is 1.0 unless scaling is required.
    """
    def lr_lambda(step):
        if step < warmup_steps:
            return min_lr + (max_lr - min_lr) * (step / warmup_steps)
        elif step < total_steps:
            decay_steps = total_steps - warmup_steps
            return min_lr + (max_lr - min_lr) * ((total_steps - step) / decay_steps)
        else:
            return min_lr  # stable after total_steps

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

from typing import Tuple, Dict, List, Optional

# Helper functions that don't need TorchScript
def create_shuffled_indices(pairs: torch.Tensor, sample_hash: int, rank: int = 0) -> List[int]:
    """Create deterministically shuffled indices based on a hash and rank."""
    total_pairs = pairs.size(0)
    
    # Create a deterministic random generator with seed + rank
    # Adding the rank ensures different devices get the same sequence
    random_gen = random.Random(sample_hash + rank * 1000000)
    
    # Generate a deterministic shuffle pattern
    shuffled_indices = list(range(total_pairs))
    random_gen.shuffle(shuffled_indices)
    
    return shuffled_indices


def rotate_indices(indices: List[int], rotation_idx: int) -> List[int]:
    """Rotate a list of indices by rotation_idx positions."""
    if not indices:
        return []
    
    total = len(indices)
    rotation_idx = rotation_idx % total  # Ensure we don't exceed bounds
    
    # Create a rotated list of indices
    return indices[rotation_idx:] + indices[:rotation_idx]


@torch.jit.script
def process_assay_vals_rotating(
    raw_assay_vals: torch.Tensor, 
    pad_idx: int, 
    sep_idx: int, 
    end_idx: int, 
    nprops: int,
    shuffled_indices: List[int],
    rotation_idx: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Process assay values with rotation to ensure each property gets to be first.
    Now with TorchScript compatibility.
    
    Args:
        raw_assay_vals: Raw tensor of assay values with padding
        pad_idx: Padding index value
        sep_idx: Separator index value
        end_idx: End index value
        nprops: Number of property-value pairs to include
        shuffled_indices: Pre-shuffled indices 
        rotation_idx: Current rotation index (determines which property comes first)
    """
    mask = raw_assay_vals != pad_idx
    assay_vals = raw_assay_vals[mask][1:-1]  # Remove SOS/EOS tokens
    
    # Safety check
    if assay_vals.numel() == 0:
        # Return empty tensors with proper structure
        device = raw_assay_vals.device
        dummy = torch.tensor([1, sep_idx, end_idx] + [pad_idx] * (2 * nprops), device=device)
        return dummy[:-1], dummy[1:]
    
    # Reshape into property-value pairs
    pairs = assay_vals.view(-1, 2)
    total_pairs = pairs.size(0)
    
    # Apply rotation within the TorchScript function
    total = len(shuffled_indices)
    rotation_idx = rotation_idx % total  # Ensure we don't exceed bounds
    
    # Manual rotation (avoiding tolist() which isn't supported in TorchScript)
    rotated_indices: List[int] = []
    for i in range(total):
        rotated_indices.append(shuffled_indices[(i + rotation_idx) % total])
    
    if total_pairs <= nprops:
        # Use all rotated indices
        selected_indices = rotated_indices
    else:
        # Take the first nprops indices from the rotated list
        selected_indices = rotated_indices[:nprops]
    
    # Get the selected pairs
    selected_pairs = torch.stack([pairs[i] for i in selected_indices])
    
    # Flatten the selected pairs
    selected_flat = selected_pairs.flatten()
    
    # Truncate or pad to the right size
    if len(selected_flat) > 2 * nprops:
        av_truncate = selected_flat[:2 * nprops]
    else:
        av_truncate = selected_flat
    
    # Add SOS and EOS tokens
    device = raw_assay_vals.device
    av_sos_eos = torch.cat([
        torch.tensor([sep_idx], device=device),
        av_truncate,
        torch.tensor([end_idx], device=device)
    ])
    
    # Pad to fixed length
    pad_value = float(pad_idx)
    out = F.pad(av_sos_eos, (0, nprops * 2 + 2 - av_sos_eos.size(0)), value=pad_value)
    
    # Create teacher forcing input (shifted by 1)
    tch = torch.cat([torch.tensor([1], device=device), out[:-1]])
    
    return tch, out


class RotatingModuloSequenceShiftDataset(Dataset):
    def __init__(self, path, tokenizer, nprops=5, assay_filter=None, rank=0, world_size=1):
        """
        Dataset with rotating property sampling that's deterministic across different ranks.
        
        Args:
            path: Path to data files
            tokenizer: Tokenizer for processing sequences
            nprops: Number of property-value pairs to include
            assay_filter: Optional list of property IDs to filter
            rank: Process rank for distributed training (default: 0)
            world_size: Total number of processes (default: 1)
        """
        self.nprops = nprops
        self.assay_filter = [] if assay_filter is None else assay_filter
        self.data = []
        self.cumulative_lengths = [0]
        cumulative_length = 0
        self.tokenizer = tokenizer
        self.pad_idx = tokenizer.PAD_IDX
        self.sep_idx = tokenizer.SEP_IDX
        self.end_idx = tokenizer.END_IDX
        
        # Store rank information
        self.rank = rank
        self.world_size = world_size
        
        # Dictionary to track rotation index for each sample
        self.sample_rotations = {}
        
        # Dictionary to cache shuffled indices for each sample
        self.sample_shuffled_indices = {}
        
        # Load data
        print(f"Rank {rank}: Loading data files...")
        for file_path in tqdm.tqdm(list(pathlib.Path(path).glob("*.pt"))):
            file_data = torch.load(file_path, map_location="cpu")
            self.data.append((file_data["selfies"], file_data["assay_vals"]))
            cumulative_length += file_data["selfies"].size(0)
            self.cumulative_lengths.append(cumulative_length)

    def __len__(self):
        return self.cumulative_lengths[-1] if self.cumulative_lengths else 0
    
    def _get_global_idx(self, file_idx, local_idx):
        """Convert file_idx and local_idx to a global sample index."""
        return self.cumulative_lengths[file_idx] + local_idx
    
    def _get_shuffled_indices(self, global_idx, raw_assay_vals):
        """Get or create shuffled property indices for a sample."""
        if global_idx not in self.sample_shuffled_indices:
            # First time seeing this sample - extract properties
            mask = raw_assay_vals != self.pad_idx
            assay_vals = raw_assay_vals[mask][1:-1]  # Remove SOS/EOS
            
            if assay_vals.numel() > 0:
                # Reshape into property-value pairs
                pairs = assay_vals.view(-1, 2)
                
                # Create a simple hash of the sample content
                sample_hash = int(raw_assay_vals.sum().item()) & 0xFFFFFFFF
                
                # Create shuffled indices with rank awareness
                self.sample_shuffled_indices[global_idx] = create_shuffled_indices(
                    pairs, sample_hash, self.rank
                )
            else:
                # Empty sample
                self.sample_shuffled_indices[global_idx] = []
        
        return self.sample_shuffled_indices[global_idx]
    
    def _initialize_rotation(self, global_idx):
        """Initialize rotation index for a sample if needed."""
        if global_idx not in self.sample_rotations:
            # First time seeing this sample
            self.sample_rotations[global_idx] = 0

    def __getitem__(self, idx):
        file_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        local_idx = idx - self.cumulative_lengths[file_idx]
        global_idx = self._get_global_idx(file_idx, local_idx)

        selfies_raw = self.data[file_idx][0][local_idx]
        raw_assay_vals = self.data[file_idx][1][local_idx]
        
        # Get shuffled indices for this sample
        shuffled_indices = self._get_shuffled_indices(global_idx, raw_assay_vals)
        
        # Initialize rotation index if needed
        self._initialize_rotation(global_idx)
        
        # Get current rotation index
        current_rotation = self.sample_rotations[global_idx]
        
        # If no properties, return dummy values
        if not shuffled_indices:
            device = raw_assay_vals.device
            dummy_tch = torch.tensor([1, self.sep_idx] + [self.pad_idx] * (2 * self.nprops), device=device)
            dummy_out = torch.tensor([self.sep_idx, self.end_idx] + [self.pad_idx] * (2 * self.nprops), device=device)
            return selfies_raw, dummy_tch, dummy_out
        
        # Use rotating property sampling
        tch, out = process_assay_vals_rotating(
            raw_assay_vals,
            self.pad_idx,
            self.sep_idx,
            self.end_idx,
            self.nprops,
            shuffled_indices,
            current_rotation
        )
        
        # Increment the rotation index for next time
        self.sample_rotations[global_idx] += 1
        
        # Reshuffle if we've completed a full cycle
        total_pairs = len(shuffled_indices)
        if self.sample_rotations[global_idx] % total_pairs == 0:
            # Create a simple hash of the sample content + cycle count
            sample_hash = int(raw_assay_vals.sum().item()) & 0xFFFFFFFF
            cycle = self.sample_rotations[global_idx] // total_pairs
            new_hash = sample_hash + cycle * 10000
            
            # Create new shuffled indices with rank awareness
            pairs = raw_assay_vals[raw_assay_vals != self.pad_idx][1:-1].view(-1, 2)
            self.sample_shuffled_indices[global_idx] = create_shuffled_indices(
                pairs, new_hash, self.rank
            )
        
        return selfies_raw, tch, out
    
    def reset_rotations(self):
        """Reset all rotation indices (call at the start of each epoch if desired)."""
        self.sample_rotations = {key: 0 for key in self.sample_rotations}
        # Optionally reset the shuffled indices as well
        self.sample_shuffled_indices = {}
    
    @staticmethod
    def setup_distributed(path, tokenizer, nprops=5, assay_filter=None, 
                          local_rank=0, world_size=1):
        """
        Factory method to create a dataset for distributed training with proper rank.
        
        Args:
            path: Path to data files
            tokenizer: Tokenizer for processing sequences
            nprops: Number of property-value pairs to include
            assay_filter: Optional list of property IDs to filter
            local_rank: Local process rank
            world_size: Total number of processes
            
        Returns:
            RotatingModuloSequenceShiftDataset instance configured for distributed training
        """
        return RotatingModuloSequenceShiftDataset(
            path=path,
            tokenizer=tokenizer,
            nprops=nprops,
            assay_filter=assay_filter,
            rank=local_rank,
            world_size=world_size
        )