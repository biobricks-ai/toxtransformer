# Sample Generator

Helper functions for generating distinct samples ready for multitask encoder inference.

## Overview

The `sample_generator.py` module provides utilities to create batches of molecular samples with different property combinations. This is useful for:

- Running inference on molecules with various property contexts
- Testing how property ordering affects predictions
- Generating evaluation datasets
- Creating synthetic samples for analysis

## Key Concepts

When using the multitask encoder, each sample consists of:
- **SELFIES tensor**: Encoded molecular structure `[seq_len]`
- **Properties tensor**: Property tokens `[nprops]`
- **Values tensor**: Corresponding values `[nprops]`
- **Mask tensor**: Boolean mask for valid pairs `[nprops]`

The target property is conventionally placed at the **last position** (index `nprops-1`), and the model predicts its value based on the molecular structure and the context provided by other properties.

## Functions

### 1. `generate_encoder_samples()`

Core function that generates distinct samples with random property combinations.

```python
from code.helper.sample_generator import generate_encoder_samples
import torch

# Already have encoded SELFIES
selfies_tensor = torch.tensor([1, 2, 3, ...])  # [seq_len]

# Generate samples
selfies_batch, props, vals, mask = generate_encoder_samples(
    selfies=selfies_tensor,
    nprops=5,                           # 5 properties per sample (4 context + 1 target)
    property_token=500,                 # Target property to predict
    available_properties=[100, 200, 300, 400],  # Properties to use as context
    available_values=[0, 1],            # Binary values
    num_samples=10,                     # Generate 10 distinct samples
    pad_idx=0,
    seed=42
)

# Output shapes:
# selfies_batch: [10, seq_len]
# props:         [10, 5]
# vals:          [10, 5]
# mask:          [10, 5]
```

**Parameters:**
- `selfies`: Encoded SELFIES tensor `[seq_len]`
- `nprops`: Number of properties per sample
- `property_token`: Target property token (placed at position `nprops-1`)
- `available_properties`: List of property tokens to randomly sample from
- `available_values`: List of value tokens to assign
- `num_samples`: Number of distinct samples to generate
- `pad_idx`: Padding index (default: 0)
- `seed`: Random seed for reproducibility

**Returns:**
- `selfies_batch`: `[num_samples, seq_len]` - Replicated SELFIES
- `properties_batch`: `[num_samples, nprops]` - Property tokens
- `values_batch`: `[num_samples, nprops]` - Value tokens
- `mask_batch`: `[num_samples, nprops]` - Boolean mask

### 2. `generate_encoder_samples_from_molecule()`

Convenience wrapper that handles SELFIES encoding.

```python
from code.helper.sample_generator import generate_encoder_samples_from_molecule
import cvae.tokenizer.selfies_property_val_tokenizer as spt

# Load tokenizer
tokenizer = spt.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

# Generate samples directly from SELFIES string
selfies_batch, props, vals, mask = generate_encoder_samples_from_molecule(
    selfies_str="[C][C][O]",           # SELFIES string
    nprops=5,
    property_token=500,
    tokenizer=tokenizer,
    available_properties=[100, 200, 300],  # Optional, will use defaults if None
    available_values=[0, 1],                # Optional, defaults to [0, 1]
    num_samples=20,
    device='cuda',
    seed=123
)
```

**Additional Parameters:**
- `selfies_str`: SELFIES string representation of molecule
- `tokenizer`: SelfiesPropertyValTokenizer instance
- `device`: Device to place tensors on ('cpu' or 'cuda')

If `available_properties` is `None`, the function will automatically select a subset of properties from the tokenizer (excluding the target property).

### 3. `generate_property_permutations()`

Generate samples with all possible orderings of a fixed set of properties.

```python
from code.helper.sample_generator import generate_property_permutations

# Generate all permutations of 3 properties
selfies_batch, props, vals, mask = generate_property_permutations(
    selfies=selfies_tensor,
    property_token=500,                # Target property (always last)
    other_properties=[100, 200, 300],  # 3 properties to permute
    other_values=[1, 0, 1],            # Their fixed values
    pad_idx=0
)

# Output: 6 samples (3! = 6 permutations)
# Each sample has a different ordering of the 3 properties,
# with the target property always at the end
```

**Use Cases:**
- Test sensitivity to property ordering
- Systematic exploration of property contexts
- Ensemble predictions by averaging over orderings

## Complete Workflow Example

```python
import torch
import cvae.models.multitask_encoder as mte
from code.helper.sample_generator import generate_encoder_samples_from_molecule

# Load model
model = mte.MultitaskEncoder.load('path/to/checkpoint').eval()
tokenizer = model.tokenizer

# Generate samples
selfies_str = "[C][C][C][O]"  # Propanol
property_token = 100  # Property to predict

selfies_batch, props, vals, mask = generate_encoder_samples_from_molecule(
    selfies_str=selfies_str,
    nprops=5,
    property_token=property_token,
    tokenizer=tokenizer,
    num_samples=50,
    device='cuda'
)

# Run inference
with torch.no_grad():
    selfies_batch = selfies_batch.to('cuda')
    props = props.to('cuda')
    vals = vals.to('cuda')
    mask = mask.to('cuda')

    logits = model(selfies_batch, props, vals, mask)  # [50, 5, num_classes]

    # Get predictions for target property (at position nprops-1 = 4)
    target_logits = logits[:, -1, :]  # [50, num_classes]
    target_probs = torch.softmax(target_logits, dim=-1)
    prob_of_1 = target_probs[:, 1]  # [50]

    # Aggregate predictions
    mean_prob = prob_of_1.mean().item()
    std_prob = prob_of_1.std().item()

    print(f"Predicted probability: {mean_prob:.4f} ± {std_prob:.4f}")
```

## Understanding the Data Format

### Property Token vs. Property ID

From `2_2_build_sqlite.py`, we see:
- **Property ID**: Raw database identifier (e.g., "AID1234")
- **Property Token**: Tokenizer index used in tensors (offset by `selfies_offset`)

```python
# In 2_2_build_sqlite.py:
property_token = assay_index  # Used in tensors
pytorch_id = tokenizer.assay_id_to_token_idx(assay_index)  # Full tokenizer index
```

When using this module:
- `property_token` should be the **raw assay_index** (0, 1, 2, ...)
- The tokenizer handles the offset internally

### Tensor Layout

```
Sample structure for nprops=5:

Position:  0      1      2      3      4 (nprops-1)
         [prop] [prop] [prop] [prop] [TARGET]
         [val]  [val]  [val]  [val]  [val_placeholder]
         [1]    [1]    [1]    [1]    [1]  <- mask

The model predicts the value at position 4 (the target property)
```

## Running the Examples

```bash
# Run example usage script
PYTHONPATH=./ python code/helper/sample_generator_example.py
```

This will demonstrate:
1. Basic usage with encoded tensors
2. Using the convenience wrapper
3. Property permutations
4. Complete model inference workflow

## Integration with Existing Code

This module follows the same data format as:
- `cvae/models/datasets/inmemory_target_property_values_dataset.py`
- `code/5_0_0_generate_evaluations.py`
- `code/2_2_build_sqlite.py`

The generated tensors can be directly passed to `MultitaskEncoder.forward()`:

```python
logits = model(selfies_batch, properties, values, mask)
```

## Notes

- **Padding**: Use `pad_idx=0` (consistent with the datasets)
- **Target Position**: Always at `nprops-1` by convention
- **Property Tokens**: Should be raw assay indices (not offset values)
- **Values**: Typically binary (0, 1) for classification tasks
- **Device**: Tensors can be generated on CPU and moved to GPU as needed

## See Also

- `cvae/models/datasets/inmemory_target_property_values_dataset.py` - Dataset implementation
- `code/5_0_0_generate_evaluations.py` - Evaluation workflow
- `code/2_2_build_sqlite.py` - Data preprocessing and token format
