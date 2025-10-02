import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import pathlib
from dataclasses import dataclass
from typing import Optional, Dict, Any, Tuple

import cvae.utils
from cvae.tokenizer.selfies_property_val_tokenizer import SelfiesPropertyValTokenizer


@dataclass
class MultitaskEncoderConfig:
    """Configuration class for MultitaskEncoder"""
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
    # Span masking parameters
    span_masking_rate: float = 0.15  # Probability of masking a token
    mean_span_length: float = 3.0    # Average length of masked spans
    use_span_masking: bool = True    # Enable/disable span masking


def generate_span_mask_vectorized(padding_mask: torch.Tensor, 
                                  masking_rate: float = 0.15, 
                                  mean_span_length: float = 3.0) -> torch.Tensor:
    """
    Generate span masks using fully static vectorized operations (compile-friendly).
    
    This implementation uses NO dynamic control flow - all operations have static shapes.
    
    Args:
        padding_mask: [batch_size, seq_len] boolean mask (True = valid, False = padding)
        masking_rate: Target fraction of tokens to mask
        mean_span_length: Average span length
        
    Returns:
        mask: [batch_size, seq_len] boolean mask (True = mask, False = keep)
    """
    batch_size, seq_len = padding_mask.shape
    device = padding_mask.device
    
    # Compute geometric distribution parameter
    p_end = 1.0 / mean_span_length
    
    # Initialize output mask
    span_mask = torch.zeros_like(padding_mask, dtype=torch.bool)
    
    # Fixed number of masking iterations (no early stopping)
    # This ensures static control flow
    max_spans = int(seq_len * masking_rate / mean_span_length) + 5
    
    for _ in range(max_spans):
        # Sample random starting positions for all sequences
        start_positions = torch.randint(0, seq_len, (batch_size,), device=device)
        
        # Sample span lengths from geometric distribution
        # Using exponential: Geometric(p) ≈ 1 + floor(Exp(ln(1-p)))
        uniform_samples = torch.rand(batch_size, device=device)
        span_lengths = 1 + torch.floor(-torch.log(uniform_samples) / (-torch.log(torch.tensor(1 - p_end, device=device)))).long()
        span_lengths = torch.clamp(span_lengths, min=1, max=10)  # Limit max span length
        
        # For each position in sequence, check if it should be masked
        positions = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        start_pos = start_positions.unsqueeze(1)  # [batch_size, 1]
        span_len = span_lengths.unsqueeze(1)  # [batch_size, 1]
        
        # Mask positions within the span: start_pos <= pos < start_pos + span_len
        in_span = (positions >= start_pos) & (positions < start_pos + span_len)
        
        # Only mask valid (non-padding) positions
        valid_span = in_span & padding_mask
        
        # Add to cumulative mask
        span_mask = span_mask | valid_span
    
    return span_mask


class PropertyFiLMHead(nn.Module):
    """
    Lightweight property-conditioned head.
    For each property p, predicts gamma_p, beta_p to modulate hidden states:
        h' = (1 + tanh(gamma_p)) * LN(h) + beta_p
    Then applies a shared linear classifier.
    """
    def __init__(self, hdim: int, out_size: int, use_rms_norm: bool = True, dropout: float = 0.0):
        super().__init__()
        norm_cls = RMSNorm if use_rms_norm else nn.LayerNorm
        self.norm = norm_cls(hdim)
        self.film = nn.Linear(hdim, 2 * hdim, bias=False)   # from property embedding -> (gamma, beta)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hdim, out_size, bias=False)

    def forward(self, h: torch.Tensor, prop_vec: torch.Tensor) -> torch.Tensor:
        """
        h:        [B, P, H] decoded states at property query positions
        prop_vec: [B, P, H] property embeddings for those positions
        returns:  [B, P, C]
        """
        gamma, beta = self.film(prop_vec).chunk(2, dim=-1)
        h = self.norm(h)
        h = (1 + torch.tanh(gamma)) * h + beta
        h = self.dropout(h)
        return self.classifier(h)


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
    
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False) 
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

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
    """Ultra-fast transformer block optimized for compilation"""
    
    def __init__(self, config: MultitaskEncoderConfig):
        super().__init__()
        
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
        
        # Use SwiGLU or standard FFN
        ffn_dim = config.hdim * config.ff_mult
        if config.activation == 'swiglu':
            self.ffn = SwiGLU(config.hdim, ffn_dim)
        else:
            self.ffn = nn.Sequential(
                nn.Linear(config.hdim, ffn_dim, bias=False),
                nn.GELU() if config.activation == 'gelu' else nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(ffn_dim, config.hdim, bias=False)
            )
        
        self.dropout = nn.Dropout(config.layer_dropout)
    
    def forward(self, x: torch.Tensor, cos: Optional[torch.Tensor] = None,
                sin: Optional[torch.Tensor] = None, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # Pre-norm architecture for better training dynamics
        residual = x
        x = self.norm1(x)
        x = self.attn(x, cos, sin, mask)
        x = self.dropout(x) + residual
        
        residual = x
        x = self.norm2(x)
        x = self.ffn(x)
        x = self.dropout(x) + residual
        
        return x


class FastDecoder(nn.Module):
    """Ultra-fast decoder stack optimized for torch.compile()"""
    
    def __init__(self, config: MultitaskEncoderConfig):
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList([
            FastTransformerBlock(config) for _ in range(config.num_layers)
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
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape
        
        # Get rotary embeddings if needed
        cos, sin = None, None
        if self.rope is not None:
            cos, sin = self.rope(x, T)
        
        # Apply transformer layers
        for layer in self.layers:
            if self.use_checkpoint and self.training:
                x = torch.utils.checkpoint.checkpoint(layer, x, cos, sin, mask, use_reentrant=False)
            else:
                x = layer(x, cos, sin, mask)
        
        return self.final_norm(x)


class MultitaskEncoder(nn.Module):
    """
    Ultra-fast MultitaskEncoder optimized for torch.compile().
    
    Uses modern optimizations:
    - Flash Attention 2 with SDPA fallback
    - RMSNorm instead of LayerNorm
    - SwiGLU activation
    - Rotary position embeddings
    - Fused operations
    - Pre-norm architecture
    - Span masking for SELFIES tokens
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
        
        # Store span masking parameters
        self.span_masking_rate = self.config.span_masking_rate
        self.mean_span_length = self.config.mean_span_length
        self.use_span_masking = self.config.use_span_masking
        
        # Build model components
        self._build_embeddings()
        self._build_decoder()
        self._build_classification_head()
        
        # Initialize weights optimally
        self._initialize_weights()
    
    def _build_embeddings(self):
        """Build embedding layers with proper scaling"""
        # SELFIES embedding
        self.embedding_selfies = nn.Embedding(
            num_embeddings=self.tokenizer.selfies_offset,
            embedding_dim=self.hdim,
            padding_idx=self.token_pad_idx
        )
        
        # Property and value embeddings  
        self.embedding_property_query = nn.Embedding(
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
        """Build ultra-fast decoder"""
        self.decoder = FastDecoder(self.config)

    def _build_classification_head(self):
        """Build standard classification head"""
        norm_cls = RMSNorm if self.config.use_rms_norm else nn.LayerNorm
        
        # Standard approach: norm + dropout + linear projection
        self.norm_head = norm_cls(self.hdim, eps=self.config.layer_norm_eps)
        self.dropout_head = nn.Dropout(self.config.layer_dropout)
        self.classifier = nn.Linear(self.hdim, self.output_size, bias=True)
    
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
        property_embeddings = self.embedding_property_query(properties)
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
        Ultra-fast forward pass optimized for compilation with span masking.
        """
        # Encode SELFIES tokens
        molecule_mask = selfies != self.token_pad_idx
        molecule_emb = self.embedding_selfies(selfies) * self.embed_scale

        # Apply span masking during training
        if self.training and self.use_span_masking:
            # Generate span mask using vectorized compile-friendly function
            span_mask = generate_span_mask_vectorized(
                padding_mask=molecule_mask,
                masking_rate=self.span_masking_rate,
                mean_span_length=self.mean_span_length
            )
            
            # Zero out embeddings for masked positions
            molecule_emb = molecule_emb.masked_fill(span_mask.unsqueeze(-1), 0.0)
            
            # Update the molecule mask to exclude span-masked positions
            molecule_mask = molecule_mask & ~span_mask
        
        # Apply standard dropout to non-masked, non-padded positions
        if self.training:
            dropout_mask = (torch.rand_like(selfies, dtype=torch.float) < self.config.dropout_rate) & molecule_mask
            molecule_emb = molecule_emb.masked_fill(dropout_mask.unsqueeze(-1), 0.0)
            molecule_mask = molecule_mask & ~dropout_mask
        
        # Create property-value sequence
        pv_embeddings = self.create_pv_teacher_forcing(properties, values) * self.embed_scale

        # Create property-value mask by repeating
        pv_mask = mask.repeat_interleave(2, dim=1)
        
        # Concatenate sequences
        full_sequence = torch.cat([molecule_emb, pv_embeddings], dim=1)
        full_mask = torch.cat([molecule_mask, pv_mask], dim=1)
        
        # Decode with fast transformer
        decoded = self.decoder(full_sequence, mask=full_mask)
        
        # Extract predictions at value positions (every other token after SELFIES offset)
        selfies_len = selfies.shape[1]
        decoded_values = decoded[:, selfies_len::2]  # Start after SELFIES, take every 2nd
        
        # Standard classification head
        decoded_values = self.norm_head(decoded_values)
        decoded_values = self.dropout_head(decoded_values)
        logits = self.classifier(decoded_values)  # [B, P, C]
        
        return logits
    
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
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        component_params = {
            'embedding_selfies': sum(p.numel() for p in self.embedding_selfies.parameters()),
            'embedding_property_query': sum(p.numel() for p in self.embedding_property_query.parameters()),
            'embedding_values': sum(p.numel() for p in self.embedding_values.parameters()),
            'decoder': sum(p.numel() for p in self.decoder.parameters()),
            'classifier': sum(p.numel() for p in self.classifier.parameters()),
        }
        return {'total': total_params, 'trainable': trainable_params, **component_params}

    
    def save(self, path: pathlib.Path, metadata: Optional[Dict[str, Any]] = None):
        """Enhanced save method with metadata"""
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        
        # Create directories
        cvae.utils.mk_empty_directory(path, overwrite=True)
        cvae.utils.mk_empty_directory(path / "spvt_tokenizer", overwrite=True)
        
        # Save tokenizer
        self.tokenizer.save(path / "spvt_tokenizer")
        
        # Prepare save dictionary
        save_dict = {
            'state_dict': self.state_dict(),
            'config': self.config.__dict__,
            'model_version': '2.1',  # Updated version for span masking
            'parameter_counts': self.get_num_parameters(),
            'metadata': metadata or {}
        }
        
        # Save model
        torch.save(save_dict, path / "multitask_encoder.pt")
        
        print(f"Model saved to {path}")
        print(f"Total parameters: {self.get_num_parameters()['total']:,}")
        
        return path
    
    @staticmethod
    def load(dirpath: pathlib.Path = pathlib.Path("brick/mtransform1")) -> 'MultitaskEncoder':
        """Enhanced load method"""
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
            
        else:
            # Old format fallback
            print("Warning: Loading old format checkpoint. Using default configuration.")
            config = MultitaskEncoderConfig()
            model = MultitaskEncoder(tokenizer=tokenizer, config=config)
            
            # Try to load state dict, allowing for missing keys
            model.load_state_dict(checkpoint, strict=False)
        
        model.eval()
        return model


def create_multitask_encoder(tokenizer: SelfiesPropertyValTokenizer,
                            model_size: str = 'base',
                            **custom_config) -> MultitaskEncoder:
    """Create model with optimized configurations for maximum speed"""
    
    configs = {
        'small': MultitaskEncoderConfig(
            hdim=256, nhead=8, num_layers=4, ff_mult=3,
            use_flash_attention=True, use_rms_norm=True, 
            activation='swiglu', use_rotary_embeddings=True,
            use_span_masking=True, span_masking_rate=0.15, mean_span_length=3.0
        ),
        'base': MultitaskEncoderConfig(
            hdim=512, nhead=16, num_layers=6, ff_mult=3,
            use_flash_attention=True, use_rms_norm=True,
            activation='swiglu', use_rotary_embeddings=True,
            use_span_masking=True, span_masking_rate=0.15, mean_span_length=3.0
        ),
        'large': MultitaskEncoderConfig(
            hdim=768, nhead=24, num_layers=12, ff_mult=3,
            use_flash_attention=True, use_rms_norm=True,
            activation='swiglu', use_rotary_embeddings=True,
            use_gradient_checkpointing=True,
            use_span_masking=True, span_masking_rate=0.15, mean_span_length=3.0
        ),
        'ultra': MultitaskEncoderConfig(
            hdim=1024, nhead=32, num_layers=16, ff_mult=3,
            use_flash_attention=True, use_rms_norm=True,
            activation='swiglu', use_rotary_embeddings=True,
            use_gradient_checkpointing=True,
            use_span_masking=True, span_masking_rate=0.15, mean_span_length=3.0
        ),
        'custom': MultitaskEncoderConfig(**custom_config)
    }
    
    if model_size not in configs:
        raise ValueError(f"Model size '{model_size}' not supported. Choose from: {list(configs.keys())}")
    
    return MultitaskEncoder(tokenizer, configs[model_size])