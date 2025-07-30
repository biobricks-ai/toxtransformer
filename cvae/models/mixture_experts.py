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
            num_layers=1,  # Optional: smaller than experts
            dropout_rate=0.0,  # Optional: no dropout for gating
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
                value_positions = prop_positions[prop_indices] + 1  # Shift to value positions
                
                # Apply expert output only at value positions
                # boosting_output[batch_indices, value_positions] += expert_output[batch_indices, value_positions]
                current_output[batch_indices, value_positions] = expert_output[batch_indices, value_positions]

        return boosting_output

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
