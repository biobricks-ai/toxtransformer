import pathlib
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import logging
import random
from collections import defaultdict

from cvae.models.multitask_transformer import (
    PositionalEncoding,
    generate_custom_subsequent_mask,
    MultitaskTransformer,
    SelfiesPropertyValTokenizer
)
import cvae.models.multitask_transformer as mt
import cvae.utils

class MoE_Boosting(nn.Module):
    """
    Boosting-style Mixture of Experts.
    Uses a frozen base_model and a trainable new_model to learn the residual.
    """

    def __init__(self, base_model: nn.Module, new_model: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.new_model = new_model

        # Freeze the base model parameters
        for param in self.base_model.parameters():
            param.requires_grad = False
    
    def forward(self, inp: torch.Tensor, teach_forcing: torch.Tensor, property_tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with boosting: base + residual.
        Assumes new_model accepts property_tokens while base_model does not.
        """
        with torch.no_grad():
            base_output = self.base_model(inp, teach_forcing)

        new_output = self.new_model(inp, teach_forcing, property_tokens)

        return base_output + new_output
