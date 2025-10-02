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


class TokenEmbedding(nn.Module):

    def __init__(self, vocab_size, hdim=256):
        super().__init__()
        self.hdim = hdim
        self.embedding = nn.Embedding(vocab_size, hdim)
        self.embedding_norm = nn.LayerNorm(hdim)

    def forward(self, input):
        embedded = self.embedding(input)
        return self.embedding_norm(embedded)

class ToxTransformer(nn.Module):

    def __init__(self, tokenizer):
        super().__init__()
        self.tokenizer = tokenizer
        self.num_tasks = len(tokenizer.assay_indexes())
        self.mintoken = min(tokenizer.assay_indexes().values())

        expert_hdim=32
        router = ContiguousRouting(0, self.num_tasks)
        self.expert_generator = lambda : ToxExpert(hdim=expert_hdim, nhead=2, ff_mult=4, num_layers=2, dropout_rate=0.1, output_size=2)
        self.shared_base = MultitaskTransformer(tokenizer, hdim=256, nhead=8, num_layers=48, ff_mult=4, dropout_rate=0.1, output_size=expert_hdim)
        self.multitask_heads = MultitaskHeads(num_tasks=self.num_tasks, model_generator=self.expert_generator, shared_base=self.shared_base, routing_module=router)
        
        
    def forward(self, selfies, properties, values, property_mask):
        return self.multitask_heads(selfies, properties, values, property_mask)


class RoutingModule(nn.Module):
    """Base class for routing task tokens to head indices."""
    
    def forward(self, tasks):
        """Convert task tokens to head indices.
        
        Args:
            tasks: [batch_size, seq_len] tensor of task tokens
            
        Returns:
            head_indices: [batch_size, seq_len] tensor of head indices (0 to num_heads-1)
        """
        raise NotImplementedError


class ContiguousRouting(RoutingModule):
    """Routes contiguous task tokens (possibly offset) to head indices."""
    
    def __init__(self, min_task_token, num_tasks):
        super().__init__()
        self.min_task_token = min_task_token
        self.num_tasks = num_tasks
        
    def forward(self, tasks):
        # Convert task tokens to head indices: task_token - min_task_token
        head_indices = tasks - self.min_task_token
        
        # Clamp to valid range [0, num_tasks-1], set invalid to -1
        valid_mask = (head_indices >= 0) & (head_indices < self.num_tasks)
        head_indices = torch.where(valid_mask, head_indices, -1)
        
        return head_indices


class MultitaskHeads(nn.Module):

    def __init__(self, num_tasks, model_generator, shared_base: nn.Module, routing_module: RoutingModule):
        super().__init__()
        self.num_tasks = num_tasks
        self.model_generator = model_generator
        self.heads = nn.ModuleList([self.model_generator() for _ in range(num_tasks)])
        self.shared_base = shared_base
        self.routing_module = routing_module

    def forward(self, selfies, tasks, values, property_mask):
        # Forward pass through shared base
        shared_output = self.shared_base(selfies, tasks, values, property_mask)  # [batch_size, seq_len, hdim]
        
        # Use routing module to get head indices
        head_indices = self.routing_module(tasks)  # [batch_size, seq_len]
        
        # Find which heads are actually needed
        valid_head_indices = head_indices[head_indices != -1]
        if len(valid_head_indices) == 0:
            # No valid tasks, return zeros
            batch_size, seq_len = tasks.shape
            output_size = self.heads[0](shared_output[:1, :1]).shape[-1]  # Get output size from dummy call
            return torch.zeros(batch_size, seq_len, output_size, device=shared_output.device, dtype=shared_output.dtype)
        
        unique_heads_needed = torch.unique(valid_head_indices)
        
        # Run all needed heads and store in list indexed by head_idx
        max_head_idx = unique_heads_needed.max()
        task_outputs = [None] * (max_head_idx + 1)
        
        for i in range(len(unique_heads_needed)):
            head_idx = unique_heads_needed[i]
            task_outputs[head_idx] = self.heads[head_idx](shared_output)
        
        # Create output tensor with matching dtype
        batch_size, seq_len = tasks.shape
        output_size = task_outputs[unique_heads_needed[0]].shape[-1]
        output = torch.zeros(batch_size, seq_len, output_size, device=shared_output.device, dtype=task_outputs[unique_heads_needed[0]].dtype)
        
        # Fill in outputs for each head
        for i in range(len(unique_heads_needed)):
            head_idx = unique_heads_needed[i]
            mask = (head_indices == head_idx)
            output[mask] = task_outputs[head_idx][mask]
        
        return output

class ToxExpert(nn.Module):

    def __init__(self, hdim, nhead, ff_mult, num_layers, dropout_rate, output_size):
        super().__init__()
        self.hdim = hdim
        self.nhead = nhead
        self.ff_mult = ff_mult
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate
        self.output_size = output_size

        self.decoder = Decoder(
            dim=self.hdim,
            depth=num_layers,
            heads=self.nhead,
            ff_mult=ff_mult,
            rotary_pos_emb=True,
            attn_dropout=dropout_rate,
            ff_dropout=dropout_rate,
            cross_attend=False
        )

        self.classification_layers = nn.Sequential(
            nn.LayerNorm(self.hdim),
            nn.Linear(self.hdim, self.output_size)
        )

    def forward(self, base_output):
        x = self.decoder(base_output)
        x = self.classification_layers(x)
        return x


class MultitaskTransformer(nn.Module):
 
    def __init__(self, tokenizer, hdim=256, nhead=2, num_layers=4, 
        ff_mult=4, dropout_rate=0.1, output_size=None):
        
        super().__init__()

        self.output_size = len(tokenizer.value_indexes()) if output_size is None else output_size
        self.hdim = hdim
        self.nhead = nhead
        self.ff_mult = ff_mult
        self.dropout_rate = dropout_rate  # Store for saving
        self.num_layers = num_layers  # Store for saving
        self.tokenizer = tokenizer
        self.token_pad_idx = tokenizer.PAD_IDX

        # we map the minimum property token to vocab_size + 1         
        # prop_val = (prop - minprop) + (val - minval + 1) * num_props
        self.numprops = len(tokenizer.assay_indexes())
        self.minprop = min(tokenizer.assay_indexes().values())
        self.minval = min(tokenizer.value_indexes().values())
        self.pv_offset = tokenizer.vocab_size
        self.num_selfies_tokens = tokenizer.selfies_offset
        self.numvals = len(tokenizer.value_indexes())

        # Single embedding for everything: SELFIES tokens, property tokens, and property-value combinations
        self.embedding_selfies = TokenEmbedding(10 + tokenizer.selfies_offset, hdim)
        self.embedding_property = TokenEmbedding(len(tokenizer.assay_indexes()), hdim)
        self.embedding_values = TokenEmbedding(len(tokenizer.value_indexes()), hdim)

        self.encoder = Encoder(
            dim=self.hdim,
            depth=num_layers,
            heads=self.nhead,
            ff_mult=ff_mult,
            rotary_pos_emb=True,
            attn_dropout=dropout_rate,
            ff_dropout=dropout_rate
        )

        self.decoder = Decoder(
            dim=self.hdim,
            depth=num_layers,
            heads=self.nhead,
            ff_mult=ff_mult,
            rotary_pos_emb=True,
            attn_dropout=dropout_rate,
            ff_dropout=dropout_rate,
            cross_attend=True
        )

        self.classification_layers = nn.Sequential(
            nn.LayerNorm(self.hdim),
            nn.Linear(self.hdim, self.output_size)
        )

    def create_pv_teacher_forcing(self, properties, values):
        """
        Map (property, value) pairs to property-value combination tokens and interleave.
        
        Args:
            properties: [batch_size, num_props] - property token indices
            values: [batch_size, num_props] - value token indices (0 or 1)
            
        Returns:
            interleaved_tokens: [batch_size, num_props * 2, hdim] - interleaved property and value embeddings
        """
        batch_size, num_props = properties.shape

        # Get embeddings for properties and values
        property_embeddings = self.embedding_property(properties)  # [batch_size, num_props, hdim]
        value_embeddings = self.embedding_values(values)          # [batch_size, num_props, hdim]
        value_embeddings = value_embeddings + property_embeddings

        # Interleave: p1, pv1, p2, pv2, ...
        # Stack along a new dimension, then reshape to interleave
        interleaved = torch.stack([property_embeddings, value_embeddings], dim=2)  # [batch_size, num_props, 2, hdim]
        interleaved_tokens = interleaved.view(batch_size, num_props * 2, self.hdim)  # [batch_size, num_props * 2, hdim]

        return interleaved_tokens

    def forward(self, selfies, properties, values, mask):
        """
        Args:
            selfies: [batch_size, selfies_seq_len] - tokenized SELFIES molecule representation
            properties: [batch_size, num_props] - property tokens 
            values: [batch_size, num_props] - corresponding value tokens
            
        Returns:
            logits: [batch_size, num_props, output_size] - predictions for each property-value position
        """
        
        # Encode molecule (SELFIES) using standard token embeddings
        molecule_mask = selfies != self.token_pad_idx
        molecule_emb = self.embedding_selfies(selfies) * math.sqrt(self.hdim)
        molecule_encoding = self.encoder(molecule_emb, mask=molecule_mask)
        
        # Map property-value pairs to combination tokens
        pv_embeddings = self.create_pv_teacher_forcing(properties, values) * math.sqrt(self.hdim)

        # Embed property-value combinations
        # Create mask by doubling each element of property_mask
        pv_mask = mask.repeat_interleave(2, dim=1)
        
        # 50% of the time just mask the entire property-value sequence
        if random.random() < 0.5 and self.training:
            pv_mask = torch.zeros_like(pv_mask, dtype=torch.bool)
        
        # Decoder uses molecule context to process property-value combinations
        decoded = self.decoder(
            pv_embeddings,
            context=molecule_encoding,
            mask=pv_mask,
            context_mask=molecule_mask
        )

        decoded_values = decoded[:, 0::2]
        logits = self.classification_layers(decoded_values)
        return logits

    def save(self, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        cvae.utils.mk_empty_directory(path, overwrite=True)
        cvae.utils.mk_empty_directory(path / "spvt_tokenizer", overwrite=True)
        self.tokenizer.save(path / "spvt_tokenizer")
        
        # Save model state and configuration
        save_dict = {
            'state_dict': self.state_dict(),
            'config': {
                'hdim': self.hdim,
                'nhead': self.nhead,
                'ff_mult': self.ff_mult,
                'output_size': self.output_size,
                'num_layers': self.num_layers,
                'dropout_rate': self.dropout_rate
            }
        }
        torch.save(save_dict, path / "mtransformer.pt")
        return path

    @staticmethod
    def load(dirpath = pathlib.Path("brick/mtransform1")):
        dirpath = pathlib.Path(dirpath)
        tokenizer = SelfiesPropertyValTokenizer.load(dirpath / "spvt_tokenizer")
        
        # Load the saved data
        checkpoint = torch.load(dirpath / 'mtransformer.pt', map_location='cpu')
        
        # Handle both old format (just state_dict) and new format (with config)
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            # New format with configuration
            config = checkpoint['config']
            model = MultitaskTransformer(
                tokenizer=tokenizer,
                hdim=config['hdim'],
                nhead=config['nhead'],
                num_layers=config['num_layers'],
                ff_mult=config['ff_mult'],
                dropout_rate=config['dropout_rate'],
                output_size=config['output_size']
            )
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Old format - just state_dict, use defaults (may cause size mismatch)
            print("Warning: Loading old format checkpoint without configuration. This may cause size mismatches.")
            model = MultitaskTransformer(tokenizer)
            model.load_state_dict(checkpoint)
        
        model.eval()
        return model

    def get_adapter_layer(self):
        return self.classification_layers[0]  # Returns the LayerNorm layer