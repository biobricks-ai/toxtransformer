import torch
import torch.nn as nn
import pathlib
import json
import cvae.utils
from cvae.tokenizer.selfies_property_val_tokenizer import SelfiesPropertyValTokenizer

# Import the routing modules directly to avoid circular imports
class RoutingModule(nn.Module):
    """Base class for routing task tokens to head indices."""
    
    def forward(self, tasks):
        """Convert task tokens to head indices.
        
        Args:
            tasks: [batch_size, seq_len] tensor of task tokens
            
        Returns:
            head_indices: [batch_size, seq_len] tensor of head indices (0 to num_heads-1)
        """
        raise NotImplementedError

class SimpleAdapter(nn.Module):
    """Simple adapter layer with a few parameters."""
    
    def __init__(self, input_dim, hidden_dim=256, output_dim=2, dropout_rate=0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.dropout_rate = dropout_rate
        
        self.adapter = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x):
        return self.adapter(x)


class PropertyToIndexRouter(RoutingModule):
    """Router that maps property tokens directly to adapter indices."""
    
    def __init__(self, property_to_idx_mapping):
        super().__init__()
        self.property_to_idx = property_to_idx_mapping
        # Create a tensor for fast lookup
        max_prop_token = max(self.property_to_idx.keys()) if self.property_to_idx else 0
        self.lookup_table = torch.full((max_prop_token + 1,), -1, dtype=torch.long)
        for prop_token, idx in self.property_to_idx.items():
            self.lookup_table[prop_token] = idx
    
    def forward(self, tasks):
        """Convert property tokens to adapter indices.
        
        Args:
            tasks: [batch_size, seq_len] tensor of property tokens
            
        Returns:
            head_indices: [batch_size, seq_len] tensor of adapter indices (0 to num_adapters-1, or -1 for invalid)
        """
        device = tasks.device
        if self.lookup_table.device != device:
            self.lookup_table = self.lookup_table.to(device)
        
        # Clamp tasks to valid range for lookup table
        clamped_tasks = torch.clamp(tasks, 0, len(self.lookup_table) - 1)
        
        # Look up adapter indices
        head_indices = self.lookup_table[clamped_tasks]
        
        # Set invalid property tokens to -1
        invalid_mask = (tasks >= len(self.lookup_table)) | (tasks < 0)
        head_indices = torch.where(invalid_mask, -1, head_indices)
        
        return head_indices


class ImprovedMultitaskHeads(nn.Module):
    """Improved MultitaskHeads with save/load functionality."""

    def __init__(self, num_tasks, adapter_generator, shared_base: nn.Module, 
                 routing_module: RoutingModule, tokenizer: SelfiesPropertyValTokenizer):
        super().__init__()
        self.num_tasks = num_tasks
        self.adapter_generator = adapter_generator
        self.heads = nn.ModuleList([self.adapter_generator() for _ in range(num_tasks)])
        self.shared_base = shared_base
        self.routing_module = routing_module
        self.tokenizer = tokenizer

    def forward(self, selfies, tasks, values, property_mask):
        # Forward pass through shared base
        with torch.no_grad():  # Shared base is frozen
            shared_output = self.shared_base(selfies, tasks, values, property_mask)
        
        # Get the LayerNorm output from the classification layers as adapter input
        adapter_input = self.shared_base.classification_layers[0](shared_output)  # Apply LayerNorm only
        
        # Use routing module to get head indices
        head_indices = self.routing_module(tasks)  # [batch_size, seq_len]
        
        # Find which heads are actually needed
        valid_head_indices = head_indices[head_indices != -1]
        if len(valid_head_indices) == 0:
            # No valid tasks, return zeros
            batch_size, seq_len = tasks.shape
            output_size = self.heads[0](adapter_input[:1, :1]).shape[-1]  # Get output size from dummy call
            return torch.zeros(batch_size, seq_len, output_size, device=adapter_input.device, dtype=adapter_input.dtype)
        
        unique_heads_needed = torch.unique(valid_head_indices)
        
        # Run all needed heads and store in list indexed by head_idx
        max_head_idx = unique_heads_needed.max()
        task_outputs = [None] * (max_head_idx + 1)
        
        for i in range(len(unique_heads_needed)):
            head_idx = unique_heads_needed[i]
            task_outputs[head_idx] = self.heads[head_idx](adapter_input)
        
        # Create output tensor with matching dtype
        batch_size, seq_len = tasks.shape
        output_size = task_outputs[unique_heads_needed[0]].shape[-1]
        output = torch.zeros(batch_size, seq_len, output_size, device=adapter_input.device, dtype=task_outputs[unique_heads_needed[0]].dtype)
        
        # Fill in outputs for each head
        for i in range(len(unique_heads_needed)):
            head_idx = unique_heads_needed[i]
            mask = (head_indices == head_idx)
            output[mask] = task_outputs[head_idx][mask]
        
        return output

    def save(self, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)

        cvae.utils.mk_empty_directory(path, overwrite=True)
        cvae.utils.mk_empty_directory(path / "spvt_tokenizer", overwrite=True)
        self.tokenizer.save(path / "spvt_tokenizer")
        
        # Save shared base
        shared_base_path = path / "shared_base"
        self.shared_base.save(shared_base_path)
        
        # Save routing module config
        routing_config = {
            'type': type(self.routing_module).__name__,
            'property_to_idx': getattr(self.routing_module, 'property_to_idx', {})
        }
        
        # Save model state and configuration
        save_dict = {
            'state_dict': self.state_dict(),
            'config': {
                'num_tasks': self.num_tasks,
                'routing_config': routing_config
            }
        }
        torch.save(save_dict, path / "adapter_model.pt")
        return path

    @staticmethod
    def load(dirpath=pathlib.Path("brick/adapter_model"), adapter_generator=None):
        dirpath = pathlib.Path(dirpath)
        
        # Import here to avoid circular imports
        from cvae.models.multitask_transformer import MultitaskTransformer
        
        tokenizer = SelfiesPropertyValTokenizer.load(dirpath / "spvt_tokenizer")
        
        # Load shared base
        shared_base = MultitaskTransformer.load(dirpath / "shared_base")
        
        # Load the saved data
        checkpoint = torch.load(dirpath / 'adapter_model.pt', map_location='cpu')
        
        config = checkpoint['config']
        num_tasks = config['num_tasks']
        routing_config = config['routing_config']
        
        # Recreate routing module
        if routing_config['type'] == 'ContiguousRouting':
            routing_module = ContiguousRouting(
                routing_config.get('min_task_token', 0), 
                routing_config.get('num_tasks', num_tasks)
            )
        elif routing_config['type'] == 'PropertyToIndexRouter':
            routing_module = PropertyToIndexRouter(
                routing_config.get('property_to_idx', {})
            )
        else:
            raise ValueError(f"Unknown routing module type: {routing_config['type']}")
        
        # Default adapter generator if not provided
        if adapter_generator is None:
            def default_adapter_generator():
                return SimpleAdapter(shared_base.hdim, hidden_dim=256, output_dim=2, dropout_rate=0.1)
            adapter_generator = default_adapter_generator
        
        # Create model
        model = ImprovedMultitaskHeads(
            num_tasks=num_tasks,
            adapter_generator=adapter_generator,
            shared_base=shared_base,
            routing_module=routing_module,
            tokenizer=tokenizer
        )
        
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()
        return model


class AdapterTransformer(nn.Module):
    """Main adapter model that wraps MultitaskHeads with property routing."""
    
    def __init__(self, tokenizer, shared_base_path, benchmark_properties, 
                 adapter_hidden_dim=256, adapter_dropout=0.1):
        super().__init__()
        
        # Import here to avoid circular imports
        from cvae.models.multitask_transformer import MultitaskTransformer
        
        self.tokenizer = tokenizer
        self.benchmark_properties = list(benchmark_properties)
        self.num_tasks = len(self.benchmark_properties)
        
        # Load the shared base (frozen MultitaskTransformer)
        self.shared_base = MultitaskTransformer.load(shared_base_path)
        
        # Freeze the shared base
        for param in self.shared_base.parameters():
            param.requires_grad = False
        
        # Create property to index mapping
        self.property_to_idx = {prop: idx for idx, prop in enumerate(self.benchmark_properties)}
        
        # Create routing module - maps property tokens directly to adapter indices
        router = PropertyToIndexRouter(self.property_to_idx)
        
        # Create adapter generator
        def adapter_generator():
            return SimpleAdapter(
                input_dim=self.shared_base.hdim,
                hidden_dim=adapter_hidden_dim,
                output_dim=2,
                dropout_rate=adapter_dropout
            )
        
        # Create MultitaskHeads
        self.multitask_heads = ImprovedMultitaskHeads(
            num_tasks=self.num_tasks,
            adapter_generator=adapter_generator,
            shared_base=self.shared_base,
            routing_module=router,
            tokenizer=tokenizer
        )
    
    def forward(self, selfies, properties, values, mask):
        return self.multitask_heads(selfies, properties, values, mask)
    
    def save(self, path):
        if not isinstance(path, pathlib.Path):
            path = pathlib.Path(path)
        
        # Save the multitask heads (which includes everything)
        saved_path = self.multitask_heads.save(path)
        
        # Also save our specific config
        config = {
            'benchmark_properties': self.benchmark_properties,
            'property_to_idx': self.property_to_idx
        }
        
        with open(path / "adapter_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        return saved_path
    
    @staticmethod
    def load(dirpath=pathlib.Path("brick/adapter_model")):
        dirpath = pathlib.Path(dirpath)
        
        # Load adapter config
        with open(dirpath / "adapter_config.json", "r") as f:
            adapter_config = json.load(f)
        
        benchmark_properties = adapter_config['benchmark_properties']
        
        # Load tokenizer
        tokenizer = SelfiesPropertyValTokenizer.load(dirpath / "spvt_tokenizer")
        
        # Create adapter generator matching what was saved
        def adapter_generator():
            return SimpleAdapter(
                input_dim=multitask_heads.shared_base.hdim, 
                hidden_dim=24,  # Default to match typical usage
                output_dim=2, 
                dropout_rate=0.1
            )
        
        # Load multitask heads
        multitask_heads = ImprovedMultitaskHeads.load(dirpath, adapter_generator)
        
        # Create the main model
        model = AdapterTransformer.__new__(AdapterTransformer)
        model.tokenizer = tokenizer
        model.benchmark_properties = benchmark_properties
        model.num_tasks = len(benchmark_properties)
        model.property_to_idx = adapter_config['property_to_idx']
        model.shared_base = multitask_heads.shared_base
        model.multitask_heads = multitask_heads
        
        return model
