import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple
from torch.amp import autocast

import cvae.utils
from cvae.tokenizer.selfies_property_val_tokenizer import SelfiesPropertyValTokenizer

# We need to install timm to use DropPath
# pip install timm
from timm.models.layers import DropPath


@dataclass
class MultitaskEncoderConfig:
    """Configuration class for MultitaskEncoder with MoE support"""
    hdim: int = 128
    nhead: int = 8
    num_layers: int = 12
    ff_mult: int = 4
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    output_size: Optional[int] = None
    use_gradient_checkpointing: bool = False
    intermediate_dim: Optional[int] = None
    activation: str = 'swiglu'
    layer_norm_eps: float = 1e-5
    layer_dropout: float = 0.1
    use_flash_attention: bool = True
    use_rms_norm: bool = True
    use_rotary_embeddings: bool = True
    rope_theta: float = 10000.0
    max_seq_len: int = 4096
    drop_path_rate: float = 0.1
    
    # MoE specific parameters
    num_experts: int = 8
    expert_capacity_factor: float = 1.5  # Controls expert capacity
    top_k_experts: int = 2  # Number of experts to route to
    moe_freq: int = 2  # Apply MoE every N layers
    load_balance_loss_coef: float = 0.01  # Load balancing loss coefficient
    router_z_loss_coef: float = 0.001  # Router z-loss coefficient
    expert_dropout: float = 0.0  # Dropout within experts


class RMSNorm(nn.Module):
    """Fast RMSNorm implementation optimized for compilation"""
    
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Faster RMS computation
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x


class RotaryEmbedding(nn.Module):
    """Efficient rotary position embeddings optimized for compilation"""
    
    def __init__(self, dim: int, max_seq_len: int = 4096, theta: float = 10000.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.theta = theta
        
        # Pre-compute frequency bands
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq, persistent=False)
        
        # Pre-compute sin/cos for max sequence length
        seq = torch.arange(max_seq_len, dtype=torch.float)
        freqs = torch.outer(seq, inv_freq)
        self.register_buffer('cos_cached', torch.cos(freqs), persistent=False)
        self.register_buffer('sin_cached', torch.sin(freqs), persistent=False)
    
    def forward(self, x: torch.Tensor, seq_len: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            self.cos_cached[:seq_len].to(x.device, x.dtype),
            self.sin_cached[:seq_len].to(x.device, x.dtype)
        )


def apply_rotary_pos_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Apply rotary embeddings - optimized for compilation"""
    # Split x into two halves
    x1, x2 = x.chunk(2, dim=-1)
    
    # Apply rotation
    rotated = torch.cat([
        x1 * cos - x2 * sin,
        x1 * sin + x2 * cos
    ], dim=-1)
    
    return rotated


class SwiGLU(nn.Module):
    """SwiGLU activation function optimized for compilation"""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) 
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w1(x))
        up = self.w3(x)
        return self.w2(self.dropout(gate * up))


class MoERouter(nn.Module):
    """Top-k router for Mixture of Experts with load balancing"""
    
    def __init__(self, dim: int, num_experts: int, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(dim, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Route tokens to experts
        
        Args:
            x: [batch_size, seq_len, dim]
            
        Returns:
            expert_weights: [batch_size, seq_len, top_k] - weights for selected experts
            expert_indices: [batch_size, seq_len, top_k] - indices of selected experts
            routing_info: Dict with auxiliary losses and statistics
        """
        batch_size, seq_len, dim = x.shape
        
        # Compute router logits
        router_logits = self.router(x)  # [B, S, num_experts]
        
        # Get top-k experts
        top_k_logits, expert_indices = torch.topk(router_logits, self.top_k, dim=-1)
        
        # Compute routing weights (softmax over top-k)
        expert_weights = F.softmax(top_k_logits, dim=-1)
        
        # Compute auxiliary losses for load balancing
        routing_probs = F.softmax(router_logits, dim=-1)  # [B, S, num_experts]
        
        # Load balancing loss - encourage uniform expert usage
        expert_usage = routing_probs.mean(dim=[0, 1])  # [num_experts]
        load_balance_loss = (expert_usage * expert_usage).sum() * self.num_experts
        
        # Router z-loss - encourage sparsity
        router_z_loss = torch.logsumexp(router_logits, dim=-1).pow(2).mean()
        
        routing_info = {
            'load_balance_loss': load_balance_loss,
            'router_z_loss': router_z_loss,
            'expert_usage': expert_usage,
            'routing_probs': routing_probs
        }
        
        return expert_weights, expert_indices, routing_info


class MoELayer(nn.Module):
    """Mixture of Experts layer with top-k routing"""
    
    def __init__(self, config: MultitaskEncoderConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.top_k = config.top_k_experts
        
        # Router
        self.router = MoERouter(config.hdim, config.num_experts, config.top_k_experts)
        
        # Expert networks
        expert_dim = config.hdim * config.ff_mult
        self.experts = nn.ModuleList([
            SwiGLU(config.hdim, expert_dim, config.expert_dropout) 
            for _ in range(config.num_experts)
        ])
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass through MoE layer
        
        Args:
            x: [batch_size, seq_len, dim]
            
        Returns:
            output: [batch_size, seq_len, dim]
            aux_losses: Dictionary of auxiliary losses
        """
        batch_size, seq_len, dim = x.shape
        
        # Route tokens to experts
        expert_weights, expert_indices, routing_info = self.router(x)
        
        # Flatten for easier expert processing
        x_flat = x.view(-1, dim)  # [B*S, dim]
        expert_weights_flat = expert_weights.view(-1, self.top_k)  # [B*S, top_k]
        expert_indices_flat = expert_indices.view(-1, self.top_k)  # [B*S, top_k]
        
        # Initialize output
        output_flat = torch.zeros_like(x_flat)
        
        # Process tokens through selected experts
        for i in range(self.top_k):
            # Get tokens and weights for i-th expert choice
            expert_idx = expert_indices_flat[:, i]  # [B*S]
            expert_weight = expert_weights_flat[:, i:i+1]  # [B*S, 1]
            
            # Group tokens by expert
            for expert_id in range(self.num_experts):
                # Find tokens assigned to this expert
                mask = (expert_idx == expert_id)
                if not mask.any():
                    continue
                
                # Process tokens through expert
                tokens_for_expert = x_flat[mask]  # [num_tokens, dim]
                expert_output = self.experts[expert_id](tokens_for_expert)
                
                # Add weighted expert output back to result
                output_flat[mask] += expert_weight[mask] * expert_output
        
        # Reshape back to original shape
        output = output_flat.view(batch_size, seq_len, dim)
        
        return output, routing_info


class FastMultiHeadAttention(nn.Module):
    """Ultra-fast multi-head attention optimized for torch.compile()"""
    
    def __init__(self, dim: int, n_heads: int, dropout: float = 0.1, 
                 use_flash: bool = True, use_rope: bool = True):
        super().__init__()
        assert dim % n_heads == 0
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        self.use_flash = use_flash and hasattr(F, 'scaled_dot_product_attention')
        self.use_rope = use_rope
        
        # Fused QKV projection for efficiency
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.dropout = dropout
    
    def forward(self, x: torch.Tensor, cos: Optional[torch.Tensor] = None, 
                sin: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Compute QKV in one pass
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Reshape for multi-head attention
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2) 
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Apply rotary embeddings if available
        if self.use_rope and cos is not None and sin is not None:
            q = apply_rotary_pos_emb(q, cos, sin)
            k = apply_rotary_pos_emb(k, cos, sin)
        
        # Use Flash Attention if available, otherwise fallback to SDPA
        if self.use_flash:
            attn_mask = None
            
            if mask is not None:
                # Create key padding mask for SDPA
                # mask is [B, T] where True = attend, False = don't attend
                # Expand to [B, 1, T, T] for proper masking
                key_padding_mask = mask.unsqueeze(1).unsqueeze(1)  # [B, 1, 1, T]
                attn_mask = key_padding_mask.expand(B, 1, T, T)  # [B, 1, T, T]
            
            out = F.scaled_dot_product_attention(
                q, k, v, 
                attn_mask=attn_mask,
                dropout_p=self.dropout if self.training else 0.0,
                is_causal=True
            )
        else:
            # Manual attention computation with proper masking
            scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, T, T]
            
            # Apply causal mask (lower triangular)
            causal_mask = torch.triu(torch.ones(T, T, device=x.device), diagonal=1).bool()
            scores.masked_fill_(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
            
            # Apply padding mask if provided
            if mask is not None:
                # mask is [B, T], expand to [B, 1, 1, T] for broadcasting
                key_mask = mask.unsqueeze(1).unsqueeze(2)  # [B, 1, 1, T]
                # Mask out attention to padded positions
                scores.masked_fill_(~key_mask, float('-inf'))
                
                # Also mask out queries from padded positions
                query_mask = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1] 
                scores = scores.masked_fill(~query_mask, float('-inf'))
            
            attn_weights = F.softmax(scores, dim=-1)
            
            # Handle NaN from softmax of all -inf rows (padded positions)
            if mask is not None:
                # Zero out attention weights for padded query positions
                query_mask = mask.unsqueeze(1).unsqueeze(-1)  # [B, 1, T, 1]
                attn_weights = attn_weights.masked_fill(~query_mask, 0.0)
            
            if self.training:
                attn_weights = F.dropout(attn_weights, p=self.dropout)
            
            out = torch.matmul(attn_weights, v)
        
        # Reshape and project
        out = out.transpose(1, 2).contiguous().view(B, T, C)
        return self.out_proj(out)


class FastTransformerBlock(nn.Module):
    """Ultra-fast transformer block with optional MoE"""
    
    def __init__(self, config: MultitaskEncoderConfig, layer_idx: int = 0):
        super().__init__()
        self.layer_idx = layer_idx
        self.use_moe = (layer_idx % config.moe_freq == 0) and config.num_experts > 1
        
        # Choose normalization
        norm_cls = RMSNorm if config.use_rms_norm else nn.LayerNorm
        
        self.norm1 = norm_cls(config.hdim, eps=config.layer_norm_eps)
        self.attn = FastMultiHeadAttention(
            config.hdim, 
            config.nhead, 
            config.attention_dropout,
            config.use_flash_attention,
            config.use_rotary_embeddings
        )
        
        self.norm2 = norm_cls(config.hdim, eps=config.layer_norm_eps)
        
        # Use MoE or standard FFN
        if self.use_moe:
            self.ffn = MoELayer(config)
            self.has_moe = True
        else:
            ffn_dim = config.hdim * config.ff_mult
            if config.activation == 'swiglu':
                self.ffn = SwiGLU(config.hdim, ffn_dim, config.dropout_rate)
            else:
                self.ffn = nn.Sequential(
                    nn.Linear(config.hdim, ffn_dim, bias=False),
                    nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
                    nn.Dropout(config.dropout_rate),
                    nn.Linear(ffn_dim, config.hdim, bias=False)
                )
            self.has_moe = False
        
        # Split dropout for attention and FFN to be more explicit
        self.attn_dropout = nn.Dropout(config.layer_dropout)
        self.ffn_dropout = nn.Dropout(config.layer_dropout)
        
        # Stochastic Depth
        self.drop_path = DropPath(config.drop_path_rate) if config.drop_path_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor, cos: Optional[torch.Tensor] = None,
                sin: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # Pre-norm architecture for better training dynamics
        aux_losses = {}
        
        # Attention block
        residual = x
        x = self.norm1(x)
        x = self.attn(x, cos, sin, mask)
        x = self.attn_dropout(x)
        x = self.drop_path(x) + residual
        
        # FFN block (potentially MoE)
        residual = x
        x = self.norm2(x)
        
        if self.has_moe:
            x, moe_aux_losses = self.ffn(x)
            aux_losses.update({f'layer_{self.layer_idx}_{k}': v for k, v in moe_aux_losses.items()})
        else:
            x = self.ffn(x)
        
        x = self.ffn_dropout(x)
        x = self.drop_path(x) + residual
        
        return x, aux_losses


class FastDecoder(nn.Module):
    """Ultra-fast decoder stack with MoE support"""
    
    def __init__(self, config: MultitaskEncoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            FastTransformerBlock(config, layer_idx=i) for i in range(config.num_layers)
        ])
        
        # Rotary embeddings if enabled
        if config.use_rotary_embeddings:
            self.rope = RotaryEmbedding(
                config.hdim // config.nhead,
                config.max_seq_len, 
                config.rope_theta
            )
        else:
            self.rope = None
        
        # Final norm
        norm_cls = RMSNorm if config.use_rms_norm else nn.LayerNorm
        self.final_norm = norm_cls(config.hdim, eps=config.layer_norm_eps)
        
        # Gradient checkpointing
        self.use_checkpoint = config.use_gradient_checkpointing
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, T, C = x.shape
        
        # Get rotary embeddings if needed
        cos, sin = None, None
        if self.rope is not None:
            cos, sin = self.rope(x, T)
        
        # Apply transformer layers and collect auxiliary losses
        all_aux_losses = {}
        
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                # Note: checkpointing with aux losses is tricky, simplified here
                x, aux_losses = layer(x, cos, sin, mask)
            else:
                x, aux_losses = layer(x, cos, sin, mask)
            
            # Accumulate auxiliary losses
            all_aux_losses.update(aux_losses)
        
        x = self.final_norm(x)
        return x, all_aux_losses


class MultitaskEncoder(nn.Module):
    """
    Ultra-fast MultitaskEncoder with Mixture of Experts support.
    
    Uses modern optimizations including MoE layers for improved capacity
    while maintaining the same interface as the original model.
    """
    
    def __init__(self, tokenizer: SelfiesPropertyValTokenizer, 
                 config: Optional[MultitaskEncoderConfig] = None):
        super().__init__()
        
        # Use provided config or create default
        self.config = config or MultitaskEncoderConfig()
        self.tokenizer = tokenizer
        
        # Cache important tokenizer properties
        self.hdim = self.config.hdim
        self.token_pad_idx = tokenizer.PAD_IDX
        
        # Set output size
        self.output_size = (len(tokenizer.value_indexes()) 
                           if self.config.output_size is None 
                           else self.config.output_size)
        
        # Build model components
        self._build_embeddings()
        self._build_decoder()
        self._build_classification_head()
        self._build_property_adapter()
        
        # Initialize weights optimally
        self._initialize_weights()
    
    def _build_embeddings(self):
        """Build embedding layers with proper scaling"""
        # SELFIES embedding
        self.embedding_selfies = nn.Embedding(
            num_embeddings=10 + self.tokenizer.selfies_offset,
            embedding_dim=self.hdim,
            padding_idx=self.token_pad_idx
        )
        
        # Property and value embeddings  
        self.embedding_property = nn.Embedding(
            num_embeddings=len(self.tokenizer.assay_indexes()),
            embedding_dim=self.hdim
        )
        
        self.embedding_values = nn.Embedding(
            num_embeddings=len(self.tokenizer.value_indexes()),
            embedding_dim=self.hdim
        )
        
        # Embedding scaling factor
        self.embed_scale = math.sqrt(self.hdim)
    
    def _build_decoder(self):
        """Build ultra-fast decoder with MoE support"""
        self.decoder = FastDecoder(self.config)
    
    def _build_classification_head(self):
        """Build optimized classification head"""
        norm_cls = RMSNorm if self.config.use_rms_norm else nn.LayerNorm
        self.classification_layers = nn.Sequential(
            norm_cls(self.hdim, eps=self.config.layer_norm_eps),
            nn.Linear(self.hdim, self.output_size, bias=False)
        )

    def _build_property_adapter(self):
        """Lightweight FiLM-style adapter conditioned on property embeddings."""
        # Produces (gamma, beta) both in R^hdim
        self.prop_film = nn.Sequential(
            nn.Linear(self.hdim, self.hdim, bias=False),
            nn.SiLU(),
            nn.Linear(self.hdim, 2 * self.hdim, bias=False),
        )
        # Scalar gate per (batch, property) to control modulation strength in [0,1]
        self.prop_gate = nn.Sequential(
            nn.Linear(self.hdim, self.hdim // 4, bias=False),
            nn.SiLU(),
            nn.Linear(self.hdim // 4, 1, bias=False),
            nn.Sigmoid(),
        )
    
    def _initialize_weights(self):
        """Initialize weights using modern best practices"""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                # Use scaled initialization for better training
                std = (2.0 / (module.in_features + module.out_features)) ** 0.5
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if hasattr(module, 'padding_idx') and module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].fill_(0)
            elif isinstance(module, (nn.LayerNorm, RMSNorm)):
                if hasattr(module, 'weight') and module.weight is not None:
                    nn.init.ones_(module.weight)
                if hasattr(module, 'bias') and module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def create_pv_teacher_forcing(self, properties: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
        """
        Optimized teacher forcing method with better memory efficiency.
        
        Args:
            properties: [batch_size, num_props] - property token indices
            values: [batch_size, num_props] - value token indices
            
        Returns:
            interleaved_tokens: [batch_size, num_props * 2, hdim] - interleaved embeddings
        """
        batch_size, num_props = properties.shape
        
        # Get embeddings efficiently
        property_embeddings = self.embedding_property(properties)
        value_embeddings = self.embedding_values(values)
        
        # Combine property context into value embeddings
        value_embeddings = value_embeddings + property_embeddings
        
        # More efficient interleaving using stack + view
        interleaved = torch.stack([property_embeddings, value_embeddings], dim=2)
        interleaved_tokens = interleaved.view(batch_size, num_props * 2, self.hdim)
        
        return interleaved_tokens

    def forward(self, selfies: torch.Tensor, properties: torch.Tensor, 
                values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Ultra-fast forward pass optimized for compilation with MoE support.
        
        Returns logits as before, but internally uses MoE for improved capacity.
        The interface remains exactly the same for backward compatibility.
        """
        # Encode SELFIES tokens
        molecule_mask = selfies != self.token_pad_idx
        molecule_emb = self.embedding_selfies(selfies) * self.embed_scale
        
        # if training then randomly mask dropout tokens
        if self.training and self.config.dropout_rate > 0.0:
            dropout_mask = (torch.rand_like(selfies, dtype=torch.float) < self.config.dropout_rate) & molecule_mask
            molecule_emb = molecule_emb.masked_fill(dropout_mask.unsqueeze(-1), 0.0)
            molecule_mask = molecule_mask & ~dropout_mask
        
        # Create property-value sequence
        pv_embeddings = self.create_pv_teacher_forcing(properties, values) * self.embed_scale
        
        # Create property-value mask by repeating
        pv_mask = mask.repeat_interleave(2, dim=1)

        # 50% of the time just mask all property values during training
        if self.training:
            # Use bernoulli sampling instead of rand + comparison
            should_mask = torch.bernoulli(torch.full((1,), 0.5, device=pv_mask.device)).bool()
            # Apply masking conditionally using torch.where
            pv_mask[:, 1::2] = torch.where(
                should_mask, 
                torch.zeros_like(pv_mask[:, 1::2]), 
                pv_mask[:, 1::2]
            )
        
        # Concatenate sequences
        full_sequence = torch.cat([molecule_emb, pv_embeddings], dim=1)
        full_mask = torch.cat([molecule_mask, pv_mask], dim=1)
        
        # Decode with fast transformer (now with MoE support)
        decoded, aux_losses = self.decoder(full_sequence, mask=full_mask)
        
        # Store auxiliary losses for potential use in training
        if self.training:
            self._last_aux_losses = aux_losses
        
        # Extract property-value positions for prediction
        mol_seq_len = selfies.size(1)
        decoded_pv = decoded[:, mol_seq_len:]  # [B, P*2, hdim]
        decoded_values = decoded_pv[:, 0::2]   # [B, P, hdim] - value positions only
        
        # Apply property-conditioned FiLM modulation
        batch_size, num_props = properties.shape
        prop_emb = self.embedding_property(properties)  # [B, P, hdim]
        
        # Generate FiLM parameters
        film_params = self.prop_film(prop_emb)  # [B, P, 2*hdim]
        gamma, beta = film_params.chunk(2, dim=-1)  # [B, P, hdim] each
        
        # Generate gating strength
        gate = self.prop_gate(prop_emb)  # [B, P, 1]
        
        # Apply FiLM modulation with gating
        modulated = gamma * decoded_values + beta  # [B, P, hdim]
        decoded_values = decoded_values + gate * (modulated - decoded_values)
        
        # Final predictions
        logits = self.classification_layers(decoded_values)  # [B, P, output_size]
        return logits
    
    def get_auxiliary_losses(self) -> Dict[str, torch.Tensor]:
        """
        Get auxiliary losses from MoE layers for training.
        Should be called after forward pass during training.
        """
        if hasattr(self, '_last_aux_losses'):
            return self._last_aux_losses
        return {}
    
    def compute_total_loss(self, main_loss: torch.Tensor) -> torch.Tensor:
        """
        Compute total loss including MoE auxiliary losses.
        
        Args:
            main_loss: Main task loss (e.g., classification loss)
            
        Returns:
            total_loss: Main loss + weighted auxiliary losses
        """
        total_loss = main_loss
        aux_losses = self.get_auxiliary_losses()
        
        for name, aux_loss in aux_losses.items():
            if 'load_balance_loss' in name:
                total_loss = total_loss + self.config.load_balance_loss_coef * aux_loss
            elif 'router_z_loss' in name:
                total_loss = total_loss + self.config.router_z_loss_coef * aux_loss
        
        return total_loss
    
    def _validate_inputs(self, selfies: torch.Tensor, properties: torch.Tensor,
                        values: torch.Tensor, mask: torch.Tensor):
        """Validate input tensors"""
        assert selfies.dim() == 2, f"Expected 2D selfies tensor, got {selfies.dim()}D"
        assert properties.dim() == 2, f"Expected 2D properties tensor, got {properties.dim()}D"
        assert values.dim() == 2, f"Expected 2D values tensor, got {values.dim()}D"
        assert mask.dim() == 2, f"Expected 2D mask tensor, got {mask.dim()}D"
        
        assert properties.shape == values.shape == mask.shape, \
            "Properties, values, and mask must have same shape"
        
        assert selfies.size(0) == properties.size(0), \
            "Batch sizes must match across all inputs"
    
    def get_output_shape_info(self, batch_size: int, num_properties: int) -> Dict[str, Tuple[int, ...]]:
        """Get information about output shapes"""
        return {
            'logits': (batch_size, num_properties, self.output_size),
            'description': {
                'logits': 'Predictions for each property at value positions',
                'batch_dim': 'First dimension indexes samples in batch',
                'property_dim': 'Second dimension indexes properties for each sample', 
                'output_dim': f'Third dimension has {self.output_size} classes (usually 2 for binary classification)'
            }
        }
    
    def get_num_parameters(self) -> Dict[str, int]:
        """Get parameter counts by component including MoE experts"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        # Count MoE parameters separately
        moe_params = 0
        router_params = 0
        expert_params = 0
        
        for name, module in self.named_modules():
            if isinstance(module, MoELayer):
                moe_layer_params = sum(p.numel() for p in module.parameters())
                moe_params += moe_layer_params
                
                router_layer_params = sum(p.numel() for p in module.router.parameters())
                router_params += router_layer_params
                
                expert_layer_params = sum(p.numel() for p in module.experts.parameters())
                expert_params += expert_layer_params
        
        component_params = {
            'embedding_selfies': sum(p.numel() for p in self.embedding_selfies.parameters()),
            'embedding_property': sum(p.numel() for p in self.embedding_property.parameters()),
            'embedding_values': sum(p.numel() for p in self.embedding_values.parameters()),
            'decoder': sum(p.numel() for p in self.decoder.parameters()),
            'classification_layers': sum(p.numel() for p in self.classification_layers.parameters()),
            'property_adapter_film': sum(p.numel() for p in self.prop_film.parameters()),
            'property_adapter_gate': sum(p.numel() for p in self.prop_gate.parameters()),
            'moe_total': moe_params,
            'moe_routers': router_params,
            'moe_experts': expert_params,
        }
        
        return {
            'total': total_params,
            'trainable': trainable_params,
            **component_params
        }
    
    def get_moe_stats(self) -> Dict[str, Any]:
        """Get statistics about MoE usage and expert utilization"""
        if not hasattr(self, '_last_aux_losses'):
            return {'error': 'No auxiliary losses available. Run forward pass first.'}
        
        aux_losses = self._last_aux_losses
        stats = {
            'num_moe_layers': sum(1 for layer in self.decoder.layers if hasattr(layer, 'has_moe') and layer.has_moe),
            'total_experts': self.config.num_experts,
            'top_k_experts': self.config.top_k_experts,
            'moe_frequency': self.config.moe_freq,
        }
        
        # Collect expert usage statistics
        expert_usage_stats = []
        for name, value in aux_losses.items():
            if 'expert_usage' in name:
                expert_usage_stats.append(value.cpu().numpy())
        
        if expert_usage_stats:
            import numpy as np
            avg_usage = np.mean(expert_usage_stats, axis=0)
            stats.update({
                'expert_usage_mean': avg_usage.tolist(),
                'expert_usage_std': np.std(expert_usage_stats, axis=0).tolist(),
                'expert_usage_balance': 1.0 / (np.var(avg_usage) + 1e-8),  # Higher = more balanced
            })
        
        return stats
    
    def save(self, path: pathlib.Path, metadata: Optional[Dict[str, Any]] = None):
        """Enhanced save method with MoE metadata"""
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        
        # Create directories
        cvae.utils.mk_empty_directory(path, overwrite=True)
        cvae.utils.mk_empty_directory(path / "spvt_tokenizer", overwrite=True)
        
        # Save tokenizer
        self.tokenizer.save(path / "spvt_tokenizer")
        
        # Enhanced metadata with MoE information
        enhanced_metadata = {
            'model_type': 'MultitaskEncoder_MoE',
            'moe_config': {
                'num_experts': self.config.num_experts,
                'top_k_experts': self.config.top_k_experts,
                'moe_freq': self.config.moe_freq,
                'expert_capacity_factor': self.config.expert_capacity_factor,
                'load_balance_loss_coef': self.config.load_balance_loss_coef,
                'router_z_loss_coef': self.config.router_z_loss_coef,
            },
            'moe_layers': sum(1 for layer in self.decoder.layers if hasattr(layer, 'has_moe') and layer.has_moe),
        }
        
        if metadata:
            enhanced_metadata.update(metadata)
        
        # Prepare save dictionary
        save_dict = {
            'state_dict': self.state_dict(),
            'config': self.config.__dict__,
            'model_version': '2.0_MoE',
            'parameter_counts': self.get_num_parameters(),
            'metadata': enhanced_metadata
        }
        
        # Save model
        torch.save(save_dict, path / "multitask_encoder.pt")
        
        print(f"MoE MultitaskEncoder saved to {path}")
        print(f"Total parameters: {self.get_num_parameters()['total']:,}")
        print(f"MoE parameters: {self.get_num_parameters()['moe_total']:,}")
        print(f"Number of experts: {self.config.num_experts}")
        print(f"MoE layers: {enhanced_metadata['moe_layers']}/{self.config.num_layers}")
        
        return path
    
    @staticmethod
    def load(dirpath: pathlib.Path = pathlib.Path("brick/mtransform1")) -> 'MultitaskEncoder':
        """Enhanced load method with MoE support"""
        dirpath = pathlib.Path(dirpath)
        
        # Load tokenizer
        tokenizer = SelfiesPropertyValTokenizer.load(dirpath / "spvt_tokenizer")
        
        # Try to load new model first, then fall back to older versions
        model_paths = [
            dirpath / 'multitask_encoder.pt',
            dirpath / 'improved_mtransformer.pt',
            dirpath / 'mtransformer.pt'
        ]
        
        checkpoint = None
        for model_path in model_paths:
            if model_path.exists():
                checkpoint = torch.load(model_path, map_location='cpu')
                print(f"Loading model from {model_path}")
                break
        
        if checkpoint is None:
            raise FileNotFoundError(f"No model checkpoint found in {dirpath}")
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict) and 'config' in checkpoint:
            # New format with configuration
            config_dict = checkpoint['config']
            config = MultitaskEncoderConfig(**config_dict)
            
            model = MultitaskEncoder(tokenizer=tokenizer, config=config)
            model.load_state_dict(checkpoint['state_dict'], strict=False)
            
            if 'model_version' in checkpoint:
                print(f"Loaded model version: {checkpoint['model_version']}")
                
            # Print MoE information if available
            if 'MoE' in checkpoint.get('model_version', ''):
                metadata = checkpoint.get('metadata', {})
                moe_config = metadata.get('moe_config', {})
                print(f"MoE Configuration:")
                print(f"  - Experts: {moe_config.get('num_experts', 'Unknown')}")
                print(f"  - Top-k: {moe_config.get('top_k_experts', 'Unknown')}")
                print(f"  - MoE frequency: {moe_config.get('moe_freq', 'Unknown')}")
                print(f"  - MoE layers: {metadata.get('moe_layers', 'Unknown')}")
            
        else:
            # Old format fallback - convert to MoE with default settings
            print("Warning: Loading old format checkpoint. Converting to MoE with default configuration.")
            config = MultitaskEncoderConfig()
            model = MultitaskEncoder(tokenizer=tokenizer, config=config)
            
            # Try to load state dict, allowing for missing keys (MoE components will be randomly initialized)
            model.load_state_dict(checkpoint, strict=False)
            print("Note: MoE components initialized with random weights.")
        
        model.eval()
        return model

    def enable_moe_logging(self, enable: bool = True):
        """Enable/disable detailed MoE logging for debugging"""
        self._moe_logging_enabled = enable
        
        if enable:
            # Register hooks to track expert usage
            def log_expert_usage(module, input, output):
                if isinstance(module, MoERouter) and self.training:
                    expert_weights, expert_indices, routing_info = output
                    usage = routing_info['expert_usage']
                    print(f"Expert usage: {usage.cpu().numpy()}")
            
            for name, module in self.named_modules():
                if isinstance(module, MoERouter):
                    module.register_forward_hook(log_expert_usage)
    
    def freeze_non_moe_params(self):
        """Freeze all parameters except MoE components for fine-tuning"""
        for name, param in self.named_parameters():
            if 'experts' not in name and 'router' not in name:
                param.requires_grad = False
            else:
                param.requires_grad = True
        
        print("Frozen all parameters except MoE components")
    
    def unfreeze_all_params(self):
        """Unfreeze all parameters"""
        for param in self.parameters():
            param.requires_grad = True
        
        print("Unfrozen all parameters")