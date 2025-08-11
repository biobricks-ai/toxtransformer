import pathlib, torch, torch.nn as nn, torch.nn.functional as F
from cvae.models.multitask_transformer import MultitaskTransformer, SelfiesPropertyValTokenizer
import cvae.models.multitask_transformer as mt
import cvae.utils
import json
import logging
import torch
import random
from collections import defaultdict
import shutil

import torch.nn as nn
from cvae.models.multitask_transformer import MultitaskTransformer, TokenEmbedding, SelfiesPropertyValTokenizer


import torch
import abc
import torch.nn as nn
import torch.nn.functional as F


class PropertyRouter(nn.Module):
    """
    Simple property-to-expert router.
    Maps property tokens to expert routing weights via embedding.
    """
    
    def __init__(self, vocab_size: int, num_experts: int, hdim: int = 64, dropout_rate: float = 0.1):
        super().__init__()
        
        self.num_experts = num_experts
        
        # Property embedding
        self.property_embedding = nn.Embedding(vocab_size, hdim)
        
        # Direct embedding -> expert mapping
        self.to_experts = nn.Linear(hdim, num_experts)
        self.dropout = nn.Dropout(dropout_rate)
        
        # Initialize with small weights
        nn.init.normal_(self.property_embedding.weight, mean=0, std=0.1)
        nn.init.xavier_uniform_(self.to_experts.weight, gain=0.1)
        nn.init.constant_(self.to_experts.bias, 0)
    
    def forward(self, property_tokens):
        """
        Args:
            property_tokens: [B, num_props] - property token IDs
            
        Returns:
            torch.Tensor: [B, num_props, num_experts] - routing logits for each property
        """
        # Embed properties: [B, num_props, hdim]
        prop_embeddings = self.property_embedding(property_tokens)
        prop_embeddings = self.dropout(prop_embeddings)
        
        # Map to expert scores: [B, num_props, num_experts]
        expert_logits = self.to_experts(prop_embeddings)
        
        return expert_logits

class MoETaskBoosting(nn.Module, abc.ABC):
    """
    Abstract base class for task-specific boosting on top of existing models.
    Subclasses must implement the boosting logic for their specific model type.
    """
    
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model
    
    @abc.abstractmethod
    def add_boosting_expert(
        self, 
        property_token: int, 
        expert_layers: int = 1,
        expert_nhead: int = 1,
        hdim: int = 32,
        expert_dim_feedforward: int = 32,
        expert_dropout_rate: float = .1):
        """
        Add a property-specific boosting expert that uses the shared embedding.
        
        Args:
            property_token: The token ID for the property this expert specializes in
            expert_layers: Number of layers (defaults to main model layers // 2)
            expert_nhead: Number of attention heads (defaults to main model nhead // 2)
            expert_dim_feedforward: Feedforward dimension (defaults to expert_hdim * 2)
            expert_dropout_rate: Dropout rate (defaults to main model dropout)
        """
        
        # Create property-specific expert
        boosting_expert = MultitaskTransformer(
            tokenizer=self.tokenizer,
            hdim=self.hdim,
            nhead=expert_nhead,
            dim_feedforward=expert_dim_feedforward,
            num_layers=expert_layers,
            dropout_rate=expert_dropout_rate,
            output_size=self.output_size,
            shared_embedding=self.shared_embedding
        )      

        # Store expert and its config
        property_key = str(property_token)
        self.boosting_experts[property_key] = boosting_expert
        self.boosting_expert_configs[property_key] = {
            'property_token': property_token,
            'expert_layers': expert_layers,
            'expert_nhead': expert_nhead,
            'expert_dim_feedforward': expert_dim_feedforward,
            'expert_dropout_rate': expert_dropout_rate
        }
        
        return boosting_expert
    
    
    @abc.abstractmethod
    def forward(self, inp, tch, current_output):
        """
        Run boosting models only on relevant value positions efficiently.
        Assumes format: [SOS, prop1, val1, prop2, val2, ..., EOS]
        
        Args:
            inp: input tensor [B, T]
            tch: teacher forcing tensor [B, T] with [1, SOS, p1, v1, p2, v2, ..., EOS]
        
        Returns:
            torch.Tensor: boosting corrections [B, T, V]
        """
        B, T = tch.shape
        V = self.output_size

        # Get property tokens at positions (2, 4, 6, ...) - the property positions in tch
        prop_positions = torch.arange(2, T, 2, device=inp.device)
            
        # Extract property tokens: [B, num_props]
        props = tch[:, prop_positions]  # [B, num_props]
        valid_mask = (props != self.tokenizer.PAD_IDX) & (props != self.tokenizer.END_IDX)
        
        # Get unique properties present in this batch
        unique_props = torch.unique(props[valid_mask])
        
        # Filter to only properties that have boosting experts
        expert_prop_keys = set(self.boosting_experts.keys())
        active_props = [p for p in unique_props if str(p.item()) in expert_prop_keys]
        
        # Process each active property
        for prop_token in active_props:
            prop_key = str(prop_token.item())
            expert = self.boosting_experts[prop_key]
            
            # Find all positions of this property: [B, num_props] -> [B, num_props]
            prop_mask = (props == prop_token) & valid_mask
            
            if prop_mask.any():
                # Get expert output for full sequence
                expert_output = expert(inp, tch)  # [B, T, V]

                # Create position mask for this property's value positions
                # Property at position i means value at position i+1
                batch_indices, prop_indices = torch.where(prop_mask)
                value_positions = prop_positions[prop_indices] # Shift to value positions, this are the same as prop positions in the teach forcing tensor due to the left shift in output tensor
                
                # Apply expert output only at value positions
                # boosting_output[batch_indices, value_positions] += expert_output[batch_indices, value_positions]
                # update expert_output dtype to match current_output
                expert_output = expert_output.to(current_output.dtype)
                current_output[batch_indices, value_positions] += expert_output[batch_indices, value_positions]

        return current_output


class MoE(nn.Module):
    
    def __init__(
        self,
        tokenizer: SelfiesPropertyValTokenizer,
        num_experts: int = 2,
        k: int = 2,
        hdim: int = 32,
        nhead: int = 8,
        dim_feedforward: int = 32,
        noise_factor: float = 0.1,
        dropout_rate: float = 0.1,
        balance_loss_weight: float = 1.0,
        diversity_loss_weight: float = 10.0,
        expert_layers: int = 4,
        output_size: int = None,
        noise_decay_steps: int = 10000 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.hdim = hdim
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.dropout_rate = dropout_rate
        self.expert_layers = expert_layers
        self.output_size = output_size if output_size is not None else tokenizer.vocab_size

        self.num_experts = num_experts
        self.k = k
        self.noise_factor = noise_factor
        self.noise_decay_steps = noise_decay_steps

        self.balance_loss_weight = balance_loss_weight
        self.diversity_loss_weight = diversity_loss_weight
        self.balance_loss = 0.0
        self.diversity_loss = 0.0
        self.current_train_step = 0
        self.num_forward_calls = 0

        # Shared token embedding for all experts and optionally the gating network
        self.shared_embedding = TokenEmbedding(tokenizer, hdim)

        # Experts
        def mk_expert():
            return MultitaskTransformer(
                tokenizer=tokenizer,
                hdim=hdim,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                num_layers=expert_layers,
                dropout_rate=dropout_rate,
                output_size=self.output_size,
                shared_embedding=self.shared_embedding
            )

        self.experts = nn.ModuleList([mk_expert() for _ in range(num_experts)])

        # Gating network (no dropout or LoRA by default, but can be modified)
        self.gating_network = MultitaskTransformer(
            tokenizer=tokenizer,
            hdim=hdim,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            num_layers=3,  # Optional: smaller than experts
            dropout_rate=dropout_rate,  # Optional: no dropout for gating
            output_size=num_experts,
            shared_embedding=self.shared_embedding
        )

    def forward(self, input, teach_forcing=None):
        
        self.num_forward_calls += 1 if self.training else 0

        B, T = input.shape
        E = self.num_experts
        V = self.tokenizer.vocab_size

        # Compute expert outputs: List of [B, T, V]
        expert_outputs = [expert(input, teach_forcing) for expert in self.experts]
        stacked_outputs = torch.stack(expert_outputs, dim=-1)  # [B, T, V, E]

        # Compute gating scores: [B, T, E]
        gating_scores = self.gating_network(input, teach_forcing)
        if self.training and self.noise_factor > 0:
            noise = self.noise_factor - (self.noise_factor * self.current_train_step / self.noise_decay_steps)
            noise = noise if noise > 0 else 0
            gating_scores = gating_scores + torch.randn_like(gating_scores) * self.noise_factor

        # For balance loss
        soft_distribution = F.softmax(gating_scores, dim=-1)

        # Get top-2 experts: indices and values
        topk_values, topk_indices = torch.topk(gating_scores, k=self.k, dim=-1)  # [B, T, 2]

        # Softmax over top-k only
        topk_weights = F.softmax(topk_values, dim=-1)  # [B, T, 2]

        # Create one-hot mask to select top-k experts
        topk_mask = F.one_hot(topk_indices, num_classes=E).float()  # [B, T, 2, E]

        # Distribute weights into full [B, T, E] tensor
        routing_weights = (topk_weights.unsqueeze(-1) * topk_mask).sum(dim=2)  # [B, T, E]

        # Apply routing weights to expert outputs
        output = (stacked_outputs * routing_weights.unsqueeze(2)).sum(dim=-1)  # [B, T, V]

        boosted_output = self.forward_boosting(input, teach_forcing, output)

        return boosted_output

    def save(self, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        # delete the path if it exists
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        
        (path / "spvt_tokenizer").mkdir(parents=True, exist_ok=True)
        self.tokenizer.save(path / "spvt_tokenizer")
        torch.save(self.state_dict(), path / "mtransformer.pt")

        config = {
            "num_experts": self.num_experts,
            "k": self.k,
            "hdim": self.hdim,
            "nhead": self.nhead,
            "dim_feedforward": self.dim_feedforward,
            "noise_factor": self.noise_factor,
            "noise_decay_steps": self.noise_decay_steps,
            "dropout_rate": self.dropout_rate,
            "balance_loss_weight": self.balance_loss_weight,
            "diversity_loss_weight": self.diversity_loss_weight,
            "expert_layers": self.expert_layers,
            "output_size": self.output_size,
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        return path

    @staticmethod
    def load(dirpath=pathlib.Path("brick/mtransform1")):
        dirpath = pathlib.Path(dirpath)
        tokenizer = SelfiesPropertyValTokenizer.load(dirpath / "spvt_tokenizer")
        config = json.load(open(dirpath / "config.json"))
        
        model = MoE(tokenizer, **config)
        
        # Load state dict
        state_dict = torch.load(dirpath / 'mtransformer.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

class GradnormLossManager():

    def __init__(self, label_smoothing=0.0):

    def lossfn(parameters, properties, values, value_logits):
        """
        this should generate N losses where N is the number of unique properties in the properties tensor
        properties is a tensor of shape [B, T] with each index for a sample being a property token
        values is a tensor of shape [B, T] of zeroes and ones
        value_logits are predictions of shape [B, T, 2]
        """
        # Get unique properties in the batch
        unique_properties = torch.unique(properties)
        total_loss = 0.0
        
        # Calculate loss for each property separately (stratified loss)
        for prop in unique_properties:
            # Create mask for this property
            prop_mask = (properties == prop)
            
            if prop_mask.any():
                # Extract logits and targets for this property
                prop_logits = value_logits[prop_mask]  # [N_prop, 2]
                prop_values = values[prop_mask]  # [N_prop]
                
                # Calculate cross entropy loss for this property
                prop_loss = torch.nn.functional.cross_entropy(
                    prop_logits, 
                    prop_values.long(), 
                    label_smoothing=label_smoothing
                )
        
        # Average over number of properties
        return {}