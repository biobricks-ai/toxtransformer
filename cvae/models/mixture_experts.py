import pathlib, torch, torch.nn as nn, torch.nn.functional as F
from cvae.models.multitask_transformer import PositionalEncoding, generate_custom_subsequent_mask, MultitaskTransformer, SelfiesPropertyValTokenizer
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

class MoE_Property_Router(nn.Module):
    
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
        noise_decay_steps: int = 10000,
        router_hdim: int = 64
    ):
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

        # Shared token embedding for all experts
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

        # Simplified property router
        self.router = PropertyRouter(
            vocab_size=tokenizer.vocab_size,
            num_experts=num_experts,
            hdim=router_hdim,
            dropout_rate=dropout_rate
        )

        # Property-specific boosting experts
        self.boosting_experts = nn.ModuleDict()
        self.boosting_expert_configs = {}

    def extract_property_tokens(self, teach_forcing):
        """
        Extract property tokens from teach_forcing tensor.
        Assumes format: [1, SOS, p1, v1, p2, v2, ..., EOS]
        
        Args:
            teach_forcing: [B, T] tensor
            
        Returns:
            property_tokens: [B, max_props] - padded property tokens
            prop_positions: [B, max_props] - positions of properties in sequence
            valid_mask: [B, max_props] - mask for valid properties
        """
        B, T = teach_forcing.shape
        
        # Property positions are at indices (2, 4, 6, ...) in teach_forcing
        max_props = (T - 3) // 2  # Approximate max properties
        
        if max_props <= 0:
            # Handle edge case where sequence is too short
            return (
                torch.full((B, 1), self.tokenizer.PAD_IDX, device=teach_forcing.device),
                torch.zeros((B, 1), dtype=torch.long, device=teach_forcing.device),
                torch.zeros((B, 1), dtype=torch.bool, device=teach_forcing.device)
            )
        
        # Collect all possible property positions
        prop_positions_list = torch.arange(2, T, 2, device=teach_forcing.device)
        actual_max_props = len(prop_positions_list)
        
        if actual_max_props == 0:
            return (
                torch.full((B, 1), self.tokenizer.PAD_IDX, device=teach_forcing.device),
                torch.zeros((B, 1), dtype=torch.long, device=teach_forcing.device),
                torch.zeros((B, 1), dtype=torch.bool, device=teach_forcing.device)
            )
        
        # Extract property tokens: [B, actual_max_props]
        prop_positions_expanded = prop_positions_list.unsqueeze(0).expand(B, -1)
        property_tokens = torch.gather(teach_forcing, 1, prop_positions_expanded)
        
        # Create valid mask (not PAD and not EOS)
        valid_mask = (property_tokens != self.tokenizer.PAD_IDX) & (property_tokens != self.tokenizer.END_IDX)
        
        return property_tokens, prop_positions_expanded, valid_mask

    def compute_property_routing(self, teach_forcing):
        """
        Compute routing weights for each property in the batch.
        
        Args:
            teach_forcing: [B, T] tensor
            
        Returns:
            routing_weights: [B, T, E] - routing weights for each position
        """
        B, T = teach_forcing.shape
        E = self.num_experts
        
        # Extract properties
        property_tokens, prop_positions, valid_mask = self.extract_property_tokens(teach_forcing)
        
        # Get routing logits for properties: [B, num_props, E]
        property_logits = self.router(property_tokens)
        
        # Add noise during training
        if self.training and self.noise_factor > 0:
            noise_scale = max(0, self.noise_factor - (self.noise_factor * self.current_train_step / self.noise_decay_steps))
            noise = torch.randn_like(property_logits) * noise_scale
            property_logits = property_logits + noise
        
        # Initialize routing weights for all positions
        routing_weights = torch.zeros(B, T, E, device=teach_forcing.device)
        
        # For each valid property, assign routing weights to its value position
        for b in range(B):
            for p_idx in range(property_tokens.shape[1]):
                if valid_mask[b, p_idx]:
                    prop_pos = prop_positions[b, p_idx]
                    val_pos = prop_pos  # Value position in output (due to left shift)
                    
                    if val_pos < T:
                        # Get top-k experts for this property
                        prop_scores = property_logits[b, p_idx]  # [E]
                        topk_values, topk_indices = torch.topk(prop_scores, k=self.k)
                        topk_weights = F.softmax(topk_values, dim=0)
                        
                        # Assign weights
                        routing_weights[b, val_pos, topk_indices] = topk_weights
        
        # Store soft distribution for balance loss
        self.last_property_distribution = F.softmax(property_logits, dim=-1)
        self.last_valid_mask = valid_mask
        
        return routing_weights

    def compute_balance_loss(self):
        """Compute balance loss based on property-level routing decisions"""
        if not hasattr(self, 'last_property_distribution') or not hasattr(self, 'last_valid_mask'):
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Average distribution across valid properties
        valid_dist = self.last_property_distribution[self.last_valid_mask]  # [num_valid_props, E]
        
        if valid_dist.shape[0] == 0:
            return torch.tensor(0.0, device=next(self.parameters()).device)
        
        # Target is uniform distribution
        mean_dist = valid_dist.mean(dim=0)  # [E]
        target_dist = torch.full_like(mean_dist, 1.0 / self.num_experts)
        
        # KL divergence for balance
        balance_loss = F.kl_div(
            F.log_softmax(mean_dist.unsqueeze(0), dim=-1),
            target_dist.unsqueeze(0),
            reduction='batchmean'
        )
        
        return balance_loss

    def add_boosting_expert(
        self, 
        property_token: int, 
        expert_layers: int = 1,
        expert_nhead: int = 1,
        hdim: int = 32,
        expert_dim_feedforward: int = 32,
        expert_dropout_rate: float = .1
    ):
        """Add a property-specific boosting expert that uses the shared embedding."""
        
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
    
    def freeze_main_experts(self):
        """Freeze main experts and router for boosting-only training"""
        for param in self.experts.parameters():
            param.requires_grad = False
        for param in self.router.parameters():
            param.requires_grad = False
        logging.info("Froze main experts and router")
        return self

    def freeze_shared_embedding(self):
        """Freeze shared embedding to prevent catastrophic forgetting"""
        for param in self.shared_embedding.parameters():
            param.requires_grad = False
        logging.info("Froze shared embedding")
        return self

    def forward_boosting(self, inp, tch, current_output):
        """Run boosting models only on relevant value positions efficiently."""
        B, T = tch.shape
        V = self.output_size

        prop_positions = torch.arange(2, T, 2, device=inp.device)
        props = tch[:, prop_positions]
        valid_mask = (props != self.tokenizer.PAD_IDX) & (props != self.tokenizer.END_IDX)
        
        unique_props = torch.unique(props[valid_mask])
        expert_prop_keys = set(self.boosting_experts.keys())
        active_props = [p for p in unique_props if str(p.item()) in expert_prop_keys]
        
        for prop_token in active_props:
            prop_key = str(prop_token.item())
            expert = self.boosting_experts[prop_key]
            
            prop_mask = (props == prop_token) & valid_mask
            
            if prop_mask.any():
                expert_output = expert(inp, tch)
                expert_output = expert_output.to(current_output.dtype)
                
                batch_indices, prop_indices = torch.where(prop_mask)
                value_positions = prop_positions[prop_indices]
                
                current_output[batch_indices, value_positions] += expert_output[batch_indices, value_positions]

        return current_output

    def forward(self, input, teach_forcing=None):
        self.num_forward_calls += 1 if self.training else 0

        B, T = input.shape
        E = self.num_experts
        V = self.tokenizer.vocab_size

        # Compute expert outputs
        expert_outputs = [expert(input, teach_forcing) for expert in self.experts]
        stacked_outputs = torch.stack(expert_outputs, dim=-1)  # [B, T, V, E]

        # Compute property-based routing weights: [B, T, E]
        routing_weights = self.compute_property_routing(teach_forcing)

        # Apply routing weights to expert outputs
        output = (stacked_outputs * routing_weights.unsqueeze(2)).sum(dim=-1)  # [B, T, V]

        # Apply boosting experts
        boosted_output = self.forward_boosting(input, teach_forcing, output)

        # Compute balance loss for training
        if self.training:
            self.balance_loss = self.compute_balance_loss()

        return boosted_output
    
    def build_stratified_lossfn(self, label_smoothing=.001):
        ignore_index = self.tokenizer.pad_idx
        value_token_ids = torch.tensor(list(self.tokenizer.value_indexes().values()), dtype=torch.long)
        
        def lossfn(parameters, logits, output):
            batch_size, vocab_size, seq_len = logits.size()

            token_indices = torch.arange(seq_len, device=logits.device)
            is_value_position = (token_indices >= 2) & (token_indices % 2 == 0)
            is_value_position = is_value_position.unsqueeze(0).expand(batch_size, seq_len)

            is_not_pad = (output != ignore_index)
            final_mask = is_value_position & is_not_pad
            
            logits = logits.transpose(1, 2).contiguous()
            logits_selected = logits[final_mask]
            output_selected = output[final_mask]
            
            value_tokens_device = value_token_ids.to(logits.device, non_blocking=True)
            value_logits = logits_selected[:, value_tokens_device]
            binary_targets = (output_selected == value_tokens_device[1]).long()
            
            # Primary task loss
            task_loss = torch.nn.functional.cross_entropy(value_logits, binary_targets, label_smoothing=label_smoothing)
            
            # Add balance loss if training
            total_loss = task_loss
            if hasattr(parameters, 'balance_loss') and parameters.training:
                total_loss = total_loss + parameters.balance_loss_weight * parameters.balance_loss
            
            return total_loss

        return lossfn

    def save(self, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

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
            "router_hdim": self.router.property_embedding.embedding_dim,
            "boosting_expert_configs": self.boosting_expert_configs
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        return path

    @staticmethod
    def load(dirpath=pathlib.Path("brick/mtransform1")):
        dirpath = pathlib.Path(dirpath)
        tokenizer = SelfiesPropertyValTokenizer.load(dirpath / "spvt_tokenizer")
        config = json.load(open(dirpath / "config.json"))
        
        boosting_configs = config.pop("boosting_expert_configs", {})
        
        model = MoE(tokenizer, **config)
        
        # Recreate boosting experts
        for prop_key, boost_config in boosting_configs.items():
            model.add_boosting_expert(
                property_token=boost_config['property_token'],
                expert_layers=boost_config['expert_layers'],
                expert_nhead=boost_config['expert_nhead'],
                expert_dim_feedforward=boost_config['expert_dim_feedforward'],
                expert_dropout_rate=boost_config['expert_dropout_rate']
            )
        
        state_dict = torch.load(dirpath / 'mtransformer.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model

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

        # Property-specific boosting experts
        self.boosting_experts = nn.ModuleDict()  # Maps property_token -> expert
        self.boosting_expert_configs = {}  # Store configs for saving/loading

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
    
    def freeze_main_experts(self):
        """Freeze main experts and gating network for boosting-only training"""
        for param in self.experts.parameters():
            param.requires_grad = False
        for param in self.gating_network.parameters():
            param.requires_grad = False
        logging.info("Froze main experts and gating network")
        return self

    def freeze_shared_embedding(self):
        """Freeze shared embedding to prevent catastrophic forgetting"""
        for param in self.shared_embedding.parameters():
            param.requires_grad = False
        logging.info("Froze shared embedding")
        return self

    def forward_boosting(self, inp, tch, current_output):
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
    
    def build_stratified_lossfn(self,label_smoothing=.001):
        ignore_index = self.tokenizer.pad_idx
        value_token_ids = torch.tensor(list(self.tokenizer.value_indexes().values()), dtype=torch.long)
        
        def lossfn(parameters, logits, output):
            batch_size, vocab_size, seq_len = logits.size()

            token_indices = torch.arange(seq_len, device=logits.device)
            is_value_position = (token_indices >= 2) & (token_indices % 2 == 0)
            is_value_position = is_value_position.unsqueeze(0).expand(batch_size, seq_len)

            is_not_pad = (output != ignore_index)
            final_mask = is_value_position & is_not_pad
            
            logits = logits.transpose(1, 2).contiguous()
            logits_selected = logits[final_mask]
            output_selected = output[final_mask]
            
            # Move value token IDs to device
            value_tokens_device = value_token_ids.to(logits.device, non_blocking=True)
            
            # Extract logits for just the two value tokens
            value_logits = logits_selected[:, value_tokens_device]
            
            # Create binary targets (0 for first value token, 1 for second)
            binary_targets = (output_selected == value_tokens_device[1]).long()
            
            # Standard cross entropy with label smoothing on 2 classes
            loss = torch.nn.functional.cross_entropy(value_logits,binary_targets, label_smoothing=label_smoothing)
            
            return loss

        return lossfn

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
            "boosting_expert_configs": self.boosting_expert_configs
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        return path

    @staticmethod
    def load(dirpath=pathlib.Path("brick/mtransform1")):
        dirpath = pathlib.Path(dirpath)
        tokenizer = SelfiesPropertyValTokenizer.load(dirpath / "spvt_tokenizer")
        config = json.load(open(dirpath / "config.json"))
        
        # Extract boosting configs if they exist
        boosting_configs = config.pop("boosting_expert_configs", {})
        
        model = MoE(tokenizer, **config)
        
        # Recreate boosting experts
        for prop_key, boost_config in boosting_configs.items():
            model.add_boosting_expert(
                property_token=boost_config['property_token'],
                expert_hdim=boost_config['expert_hdim'],
                expert_layers=boost_config['expert_layers'],
                expert_nhead=boost_config['expert_nhead'],
                expert_dim_feedforward=boost_config['expert_dim_feedforward'],
                expert_dropout_rate=boost_config['expert_dropout_rate']
            )
        
        # Load state dict
        state_dict = torch.load(dirpath / 'mtransformer.pt', map_location='cpu', weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        return model
