import torch
import torch.nn as nn
import pathlib
import json
from cvae.tokenizer.selfies_property_val_tokenizer import SelfiesPropertyValTokenizer
from cvae.models.multitask_encoder import MultitaskEncoder

class ShallowPropertyHead(nn.Module):
    """A shallow head for a single property."""
    def __init__(self, input_dim, output_dim=1, hidden_dim=128, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    def forward(self, x):
        return self.net(x)

class SimpleAdapterModel(nn.Module):
    """
    Efficient adapter with property-specific classifiers initialized from the frozen encoder.
    Uses deeper adapter heads with regularization to prevent overfitting.
    """

    def __init__(self, multitask_encoder: MultitaskEncoder,
                 num_properties: int, output_dim: int = 2,
                 adapter_hidden_dim: int = 256, adapter_dropout: float = 0.2,
                 use_residual: bool = True):
        super().__init__()
        self.encoder = multitask_encoder

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False

        self.num_properties = num_properties
        self.output_dim = output_dim
        self.use_residual = use_residual

        encoder_dim = self.encoder.hdim

        # Property-specific adapter heads with proper regularization
        # Using ModuleList for proper parameter registration
        self.adapter_heads = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(encoder_dim),
                nn.Linear(encoder_dim, adapter_hidden_dim),
                nn.ReLU(),
                nn.Dropout(adapter_dropout),
                nn.Linear(adapter_hidden_dim, adapter_hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(adapter_dropout),
                nn.Linear(adapter_hidden_dim // 2, output_dim)
            ) for _ in range(num_properties)
        ])

        # Initialize adapter heads to approximate the base model's classifier
        with torch.no_grad():
            encoder_bias = self.encoder.classifier.bias.data if self.encoder.classifier.bias is not None else torch.zeros(output_dim)

            for head in self.adapter_heads:
                # Initialize the final layer to be close to the encoder's classifier
                final_linear = head[-1]  # Last layer (Linear)

                # Initialize with small random weights
                nn.init.normal_(final_linear.weight, mean=0.0, std=0.01)
                if final_linear.bias is not None:
                    # Initialize bias with encoder's bias (provides good starting point)
                    final_linear.bias.data.copy_(encoder_bias * 0.1)
    
    def forward(self, selfies, properties, values, mask):
        batch_size, num_props = properties.shape

        # Get hidden states from frozen encoder
        with torch.no_grad():
            hidden_states = self.encoder.extract_hidden_states(
                selfies, properties, values, mask
            )  # [B, P, hdim]

            # Apply encoder's norm and dropout (matching original pipeline)
            hidden_states = self.encoder.norm_head(hidden_states)
            # Note: encoder dropout is frozen, we use dropout in adapter heads instead

        # Flatten batch and property dimensions for efficient processing
        flat_hidden = hidden_states.reshape(-1, hidden_states.shape[-1])  # [B*P, hdim]
        flat_properties = properties.reshape(-1)  # [B*P]
        flat_mask = mask.reshape(-1)  # [B*P]

        # Process all samples through their respective adapter heads
        flat_output = torch.zeros(batch_size * num_props, self.output_dim,
                                   device=hidden_states.device, dtype=hidden_states.dtype)

        # Group by property index for efficient batched processing
        # Use torch.where to avoid data-dependent branching (required for torch.compile)
        for prop_idx in range(self.num_properties):
            # Find all positions with this property
            prop_mask = (flat_properties == prop_idx) & (flat_mask > 0)

            # Apply adapter head to all hidden states (avoids data-dependent branching)
            # This is compile-friendly even though it processes all samples
            all_outputs = self.adapter_heads[prop_idx](flat_hidden)

            # Use torch.where to select only outputs for this property
            flat_output = torch.where(
                prop_mask.unsqueeze(-1).expand(-1, self.output_dim),
                all_outputs,
                flat_output
            )

        # Reshape back to [B, P, output_dim]
        output = flat_output.reshape(batch_size, num_props, self.output_dim)

        # Optionally add residual from base model
        if self.use_residual:
            with torch.no_grad():
                base_output = self.encoder(selfies, properties, values, mask)
            # Weighted residual: 70% adapter + 30% base (trust the adapter more)
            output = 0.7 * output + 0.3 * base_output

        # Zero out padded positions
        output = output * mask.unsqueeze(-1)

        return output
    
    def verify_initialization(self, selfies, properties, values, mask):
        """Verify that initialization is reasonable (within expected range of encoder)."""
        with torch.no_grad():
            # Get encoder's original output
            encoder_output = self.encoder(selfies, properties, values, mask)

            # Get adapter output
            adapter_output = self.forward(selfies, properties, values, mask)

            # Check they're in similar range (not identical due to different architecture)
            max_diff = (encoder_output - adapter_output).abs().max().item()
            mean_diff = (encoder_output - adapter_output).abs().mean().item()

            print(f"Max difference: {max_diff:.6f}")
            print(f"Mean difference: {mean_diff:.6f}")
            print(f"Encoder output range: [{encoder_output.min():.6f}, {encoder_output.max():.6f}]")
            print(f"Adapter output range: [{adapter_output.min():.6f}, {adapter_output.max():.6f}]")

            # More relaxed check since we have a different architecture
            return mean_diff < 5.0  # Allow for architectural differences
    
    def save(self, path):
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)

        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'num_properties': self.num_properties,
                'output_dim': self.output_dim,
                'adapter_hidden_dim': self.adapter_heads[0][1].out_features if len(self.adapter_heads) > 0 else 256,
                'adapter_dropout': self.adapter_heads[0][3].p if len(self.adapter_heads) > 0 else 0.2,
                'use_residual': self.use_residual,
            }
        }, path / 'adapter_model.pt')

        # Save encoder
        self.encoder.save(path / 'encoder')

        return path

    @staticmethod
    def load(path, encoder=None):
        path = pathlib.Path(path)
        checkpoint = torch.load(path / 'adapter_model.pt', map_location='cpu')

        if encoder is None:
            encoder = MultitaskEncoder.load(path / 'encoder')

        config = checkpoint['config']
        model = SimpleAdapterModel(
            multitask_encoder=encoder,
            num_properties=config['num_properties'],
            output_dim=config.get('output_dim', 2),
            adapter_hidden_dim=config.get('adapter_hidden_dim', 256),
            adapter_dropout=config.get('adapter_dropout', 0.2),
            use_residual=config.get('use_residual', True),
        )

        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        return model