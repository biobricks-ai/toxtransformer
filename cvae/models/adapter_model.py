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
    Adapter model that takes a frozen MultitaskEncoder and adds a shallow head for each property.
    Properties are indexed 0..num_tasks-1 and routed directly to their head.
    """
    def __init__(self, multitask_encoder: MultitaskEncoder, num_properties: int, head_hidden_dim=128, head_dropout=0.1, head_output_dim=1):
        super().__init__()
        self.encoder = multitask_encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        self.num_properties = num_properties
        self.heads = nn.ModuleList([
            ShallowPropertyHead(self.encoder.hdim, output_dim=head_output_dim, hidden_dim=head_hidden_dim, dropout=head_dropout)
            for _ in range(num_properties)
        ])

    def forward(self, selfies, properties, values, mask):
        # Forward through encoder (frozen)
        with torch.no_grad():
            enc_out = self.encoder(selfies, properties, values, mask)
        # Optionally, use a norm or head from encoder if needed
        # For simplicity, use enc_out directly
        # properties: [batch_size, seq_len] with values in 0..num_properties-1
        batch_size, seq_len = properties.shape
        output = torch.zeros(batch_size, seq_len, self.heads[0].net[-1].out_features, device=enc_out.device, dtype=enc_out.dtype)
        for i in range(self.num_properties):
            mask_i = (properties == i)
            if mask_i.any():
                output[mask_i] = self.heads[i](enc_out[mask_i])
        return output

    def save(self, path):
        path = pathlib.Path(path)
        torch.save({
            'state_dict': self.state_dict(),
            'num_properties': self.num_properties,
            'head_hidden_dim': self.heads[0].net[1].out_features,
            'head_dropout': self.heads[0].net[2].p,
            'head_output_dim': self.heads[0].net[-1].out_features
        }, path / 'adapter_model.pt')
        # Save encoder separately if needed
        self.encoder.save(path / 'encoder')
        return path

    @staticmethod
    def load(path, encoder=None):
        path = pathlib.Path(path)
        checkpoint = torch.load(path / 'adapter_model.pt', map_location='cpu')
        if encoder is None:
            encoder = MultitaskEncoder.load(path / 'encoder')
        model = SimpleAdapterModel(
            multitask_encoder=encoder,
            num_properties=checkpoint['num_properties'],
            head_hidden_dim=checkpoint['head_hidden_dim'],
            head_dropout=checkpoint['head_dropout'],
            head_output_dim=checkpoint['head_output_dim']
        )
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model


## AdapterTransformer is now deprecated in favor of SimpleAdapterModel