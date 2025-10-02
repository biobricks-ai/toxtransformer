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
from entmax import sparsemax

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
        # Clamp property tokens to valid range to prevent embedding out-of-bounds
        vocab_size = self.property_embedding.num_embeddings
        property_tokens = torch.clamp(property_tokens, 0, vocab_size - 1)
        
        # Embed properties: [B, num_props, hdim]
        prop_embeddings = self.property_embedding(property_tokens)
        prop_embeddings = self.dropout(prop_embeddings)
        
        # Map to expert scores: [B, num_props, num_experts]
        expert_logits = self.to_experts(prop_embeddings)
        
        return expert_logits

class MoE(nn.Module):

    # ---- NEW: default expert kwargs used when mk_expert is not provided ----
    _DEFAULT_EXPERT_KWARGS = dict(
        hdim=128,
        nhead=8,
        num_layers=42,
        ff_mult=4,
        dropout_rate=0.1
        # output_size is set from self.output_size at init time
    )

    @staticmethod
    def default_expert_factory(tokenizer: SelfiesPropertyValTokenizer, **expert_kwargs):
        """
        Returns a callable that builds a MultitaskTransformer with the given kwargs.
        """
        def _factory():
            return MultitaskTransformer(tokenizer=tokenizer, **expert_kwargs)
        return _factory

    def __init__(
        self,
        tokenizer: SelfiesPropertyValTokenizer,
        num_experts: int = 2,
        k: int = 2,
        hdim: int = 32,
        noise_factor: float = 0.1,
        dropout_rate: float = 0.1,
        balance_loss_weight: float = 1.0,
        diversity_loss_weight: float = 10.0,
        output_size: int = None,
        noise_decay_steps: int = 10_000,
        mk_expert: callable = None,
        # ---- NEW: allow passing explicit expert kwargs so shapes persist on load ----
        expert_kwargs: dict | None = None,
    ):
        super().__init__()

        self.tokenizer = tokenizer
        self.hdim = hdim
        self.dropout_rate = dropout_rate
        self.output_size = output_size if output_size is not None else len(tokenizer.value_indexes())

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

        # ---- NEW: freeze the expert architecture used to build experts ----
        # start from defaults, override with provided dict, and ensure output_size matches MoE
        base = dict(MoE._DEFAULT_EXPERT_KWARGS)
        if expert_kwargs:
            base.update(expert_kwargs)
        base.setdefault("output_size", self.output_size)
        base["output_size"] = self.output_size  # force consistency
        self._expert_arch = base  # persisted in save()

        # ---- build experts with frozen architecture ----
        if mk_expert is None:
            mk_expert = MoE.default_expert_factory(tokenizer, **self._expert_arch)
        self.experts = nn.ModuleList([mk_expert() for _ in range(num_experts)])

        # Property router
        assay_vocab_size = len(tokenizer.assay_indexes()) + 10
        self.property_router = PropertyRouter(
            vocab_size=assay_vocab_size,
            num_experts=num_experts,
            hdim=hdim,
            dropout_rate=dropout_rate
        )

    def forward(self, selfies, properties, values, mask):
        """
        Args:
            selfies: [batch_size, selfies_seq_len] - tokenized SELFIES molecule representation
            properties: [batch_size, num_props] - property tokens 
            values: [batch_size, num_props] - corresponding value tokens
            mask: [batch_size, num_props] - mask for valid properties
            
        Returns:
            logits: [batch_size, num_props, output_size] - predictions for each property
        """
        
        self.num_forward_calls += 1 if self.training else 0
        self.current_train_step += 1 if self.training else 0

        B, num_props = properties.shape
        E = self.num_experts

        # Compute expert outputs: List of [B, num_props, output_size]
        expert_outputs = [expert(selfies, properties, values, mask) for expert in self.experts]
        stacked_outputs = torch.stack(expert_outputs, dim=-1)  # [B, num_props, output_size, E]

        # Compute routing scores using PropertyRouter: [B, num_props, E]
        routing_logits = self.property_router(properties)
        
        # Add noise during training (with decay)
        if self.training and self.noise_factor > 0:
            noise = self.noise_factor - (self.noise_factor * self.current_train_step / self.noise_decay_steps)
            noise = max(noise, 0.0)  # Ensure noise doesn't go negative
            if noise > 0:
                routing_logits = routing_logits + torch.randn_like(routing_logits) * noise

        # Apply mask to routing logits with safer masking
        masked_routing_logits = routing_logits.clone()
        if mask is not None:
            # Use a large negative value instead of -inf to avoid numerical issues
            masked_routing_logits = masked_routing_logits.masked_fill(~mask.unsqueeze(-1), -1e9)

        # Use sparsemax for sparse routing weights (more stable than entmax15)
        routing_weights = sparsemax(masked_routing_logits, dim=-1)  # [B, num_props, E]

        # For balance loss (use soft distribution for regularization)
        # Only compute on valid positions to avoid CUDA assertion errors
        soft_distribution = F.softmax(masked_routing_logits, dim=-1)

        # Apply routing weights to expert outputs
        output = (stacked_outputs * routing_weights.unsqueeze(2)).sum(dim=-1)  # [B, num_props, output_size]

        return output
    
    def freeze_router(self):
        for param in self.property_router.parameters():
            param.requires_grad = False
        return self

    def save(self, path):
        path = pathlib.Path(path)
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
        (path / "spvt_tokenizer").mkdir(parents=True, exist_ok=True)

        # tokenizer
        self.tokenizer.save(path / "spvt_tokenizer")

        # weights (keep legacy filename for compatibility)
        torch.save(self.state_dict(), path / "mtransformer.pt")

        # config (add expert_arch so load can reconstruct identical shapes)
        config = {
            "num_experts": self.num_experts,
            "k": self.k,
            "hdim": self.hdim,
            "noise_factor": self.noise_factor,
            "noise_decay_steps": self.noise_decay_steps,
            "dropout_rate": self.dropout_rate,
            "balance_loss_weight": self.balance_loss_weight,
            "diversity_loss_weight": self.diversity_loss_weight,
            "output_size": self.output_size,
            "expert_arch": self._expert_arch,  # <---- NEW
        }
        with open(path / "config.json", "w") as f:
            json.dump(config, f, indent=2)

        return path

    @staticmethod
    def load(dirpath=pathlib.Path("brick/mtransform1"), map_location=None, strict=True):
        """
        Load a saved MoE. If you move between CPU/GPU, set map_location="cpu" or "cuda".
        Set strict=False to allow minor shape/key differences.
        """
        dirpath = pathlib.Path(dirpath)

        # tokenizer / config
        tokenizer = SelfiesPropertyValTokenizer.load(dirpath / "spvt_tokenizer")
        with open(dirpath / "config.json", "r") as f:
            config = json.load(f)

        # pull out expert_arch; older checkpoints won't have it
        expert_arch = config.pop("expert_arch", None)

        # construct with identical expert architecture
        model = MoE(
            tokenizer,
            expert_kwargs=expert_arch,
            **config
        )

        # weights file (support legacy or renamed paths)
        weights_path_candidates = [
            dirpath / "mtransformer.pt",  # current/legacy name
            dirpath / "moe.pt"            # optional future name
        ]
        weights_path = next((p for p in weights_path_candidates if p.exists()), None)
        if weights_path is None:
            raise FileNotFoundError(f"No weights file found in {dirpath} among {weights_path_candidates}")

        # load
        if map_location is None:
            map_location = "cpu"
        state_dict = torch.load(weights_path, map_location=map_location)
        missing, unexpected = model.load_state_dict(state_dict, strict=strict)

        if missing or unexpected:
            logging.warning(f"While loading MoE: missing_keys={missing}, unexpected_keys={unexpected}")

        return model
