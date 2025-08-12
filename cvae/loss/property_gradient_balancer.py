import torch
import logging
from collections import defaultdict


class PropertyGradientBalancer:
    """
    Balances gradients across properties during training.
    Each distributed model instance maintains its own balancer.
    """
    
    def __init__(self, tokenizer, ema_decay=0.99, min_weight=0.1, max_weight=5.0):
        self.tokenizer = tokenizer
        self.ema_decay = ema_decay
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Track gradient magnitudes per property using EMA
        self.property_grad_magnitudes = {}
        self.property_sample_counts = {}  # Track how many samples we've seen
        
        # Get all property tokens
        self.property_tokens = list(tokenizer.assay_indexes().values())
        
        # Initialize with small positive values to avoid division by zero
        for prop in self.property_tokens:
            self.property_grad_magnitudes[prop] = 1.0
            self.property_sample_counts[prop] = 0
            
        self.total_updates = 0
        
    def update_gradient_stats(self, property_losses):
        """
        Update EMA of gradient magnitudes for each property.
        
        Args:
            property_losses: Dict mapping property_token -> loss_tensor
        """
        self.total_updates += 1
        
        for prop_token, loss_tensor in property_losses.items():
            if torch.is_tensor(loss_tensor) and loss_tensor.numel() > 0:
                loss_value = float(loss_tensor.detach().cpu())
                
                if loss_value > 0:  # Only update if we have a valid loss
                    current_mag = self.property_grad_magnitudes.get(prop_token, 1.0)
                    
                    # EMA update
                    self.property_grad_magnitudes[prop_token] = (
                        self.ema_decay * current_mag + (1 - self.ema_decay) * loss_value
                    )
                    
                    # Count samples
                    self.property_sample_counts[prop_token] += 1
    
    def get_balancing_weights(self):
        """
        Get weights to balance gradient magnitudes across properties.
        
        Returns:
            Dict mapping property_token -> weight_factor
        """
        if self.total_updates < 10:  # Use uniform weights initially
            return {prop: 1.0 for prop in self.property_tokens}
        
        # Calculate average magnitude across all properties
        active_magnitudes = [
            self.property_grad_magnitudes[prop] 
            for prop in self.property_tokens 
            if self.property_sample_counts.get(prop, 0) > 0
        ]
        
        if not active_magnitudes:
            return {prop: 1.0 for prop in self.property_tokens}
        
        avg_magnitude = sum(active_magnitudes) / len(active_magnitudes)
        
        # Compute inverse weights with smoothing and clipping
        weights = {}
        for prop in self.property_tokens:
            prop_mag = self.property_grad_magnitudes.get(prop, avg_magnitude)
            
            # Inverse weighting: properties with smaller losses get higher weights
            raw_weight = avg_magnitude / (prop_mag + 1e-8)
            
            # Clip to reasonable range
            weight = max(self.min_weight, min(self.max_weight, raw_weight))
            weights[prop] = weight
        
        return weights
    
    def get_stats(self):
        """Get current balancing statistics for logging."""
        weights = self.get_balancing_weights()
        
        stats = {
            'total_updates': self.total_updates,
            'avg_weight': sum(weights.values()) / len(weights) if weights else 1.0,
            'weight_std': torch.tensor(list(weights.values())).std().item() if weights else 0.0,
            'active_properties': len([p for p in self.property_tokens if self.property_sample_counts.get(p, 0) > 0]),
            'property_magnitudes': {k: round(v, 4) for k, v in self.property_grad_magnitudes.items()},
            'property_weights': {k: round(v, 4) for k, v in weights.items()}
        }
        
        return stats
    
    def reset(self):
        """Reset all statistics (useful for validation/testing)."""
        for prop in self.property_tokens:
            self.property_grad_magnitudes[prop] = 1.0
            self.property_sample_counts[prop] = 0
        self.total_updates = 0