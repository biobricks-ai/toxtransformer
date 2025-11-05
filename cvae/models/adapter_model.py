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
    Starts with identical performance to the base model.
    """
    
    def __init__(self, multitask_encoder: MultitaskEncoder, 
                 num_properties: int, output_dim: int = 2):
        super().__init__()
        self.encoder = multitask_encoder
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
            
        self.num_properties = num_properties
        self.output_dim = output_dim
        
        encoder_dim = self.encoder.hdim
        
        # Get the frozen encoder's classifier weights and bias
        with torch.no_grad():
            encoder_weight = self.encoder.classifier.weight.data  # [output_dim, hdim]
            encoder_bias = self.encoder.classifier.bias.data if self.encoder.classifier.bias is not None else torch.zeros(output_dim)
        
        # Initialize property-specific classifiers with encoder's weights
        # Shape: [num_properties, hdim, output_dim]
        self.classifier_weights = nn.Parameter(
            encoder_weight.T.unsqueeze(0).repeat(num_properties, 1, 1).clone()
        )
        
        # Shape: [num_properties, output_dim]
        self.classifier_bias = nn.Parameter(
            encoder_bias.unsqueeze(0).repeat(num_properties, 1).clone()
        )
        
        # Add small random perturbation to break symmetry (optional)
        with torch.no_grad():
            self.classifier_weights.add_(torch.randn_like(self.classifier_weights) * 1e-4)
            self.classifier_bias.add_(torch.randn_like(self.classifier_bias) * 1e-4)
    
    def forward(self, selfies, properties, values, mask):
        batch_size, num_props = properties.shape
        
        # Get hidden states from frozen encoder
        with torch.no_grad():
            hidden_states = self.encoder.extract_hidden_states(
                selfies, properties, values, mask
            )  # [B, P, hdim]
            
            # Apply encoder's norm and dropout (matching original pipeline)
            hidden_states = self.encoder.norm_head(hidden_states)
            hidden_states = self.encoder.dropout_head(hidden_states)
        
        # Gather the property-specific weights and biases
        weights = self.classifier_weights[properties]  # [B, P, hdim, output_dim]
        biases = self.classifier_bias[properties]     # [B, P, output_dim]
        
        # Apply property-specific linear transformation
        hidden_expanded = hidden_states.unsqueeze(-2)  # [B, P, 1, hdim]
        output = torch.matmul(hidden_expanded, weights).squeeze(-2)  # [B, P, output_dim]
        output = output + biases
        
        # Zero out padded positions
        output = output * mask.unsqueeze(-1)
        
        return output
    
    def verify_initialization(self, selfies, properties, values, mask):
        """Verify that initialization matches encoder's output."""
        with torch.no_grad():
            # Get encoder's original output
            encoder_output = self.encoder(selfies, properties, values, mask)
            
            # Get adapter output
            adapter_output = self.forward(selfies, properties, values, mask)
            
            # Check they're nearly identical
            max_diff = (encoder_output - adapter_output).abs().max().item()
            mean_diff = (encoder_output - adapter_output).abs().mean().item()
            
            print(f"Max difference: {max_diff:.6f}")
            print(f"Mean difference: {mean_diff:.6f}")
            
            return max_diff < 1e-3
    
    def save(self, path):
        path = pathlib.Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        torch.save({
            'state_dict': self.state_dict(),
            'config': {
                'num_properties': self.num_properties,
                'output_dim': self.output_dim,
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
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        
        return model