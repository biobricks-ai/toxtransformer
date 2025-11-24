"""
Example usage of sample_generator.py

This script demonstrates how to use the sample generation functions
to create batches ready for multitask encoder inference.
"""

import torch
import cvae.tokenizer.selfies_property_val_tokenizer as spt
from code.helper.sample_generator import (
    generate_encoder_samples,
    generate_encoder_samples_from_molecule,
    generate_property_permutations
)


def example_basic_usage():
    """Basic usage with pre-encoded SELFIES tensor."""
    print("=" * 80)
    print("Example 1: Basic usage with encoded SELFIES tensor")
    print("=" * 80)

    # Load tokenizer
    tokenizer = spt.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

    # Encode a SELFIES string
    selfies_str = "[C][C][C][O]"  # Propanol
    encoded_selfies = tokenizer.selfies_tokenizer.encode_selfies(selfies_str)
    selfies_tensor = torch.tensor(encoded_selfies, dtype=torch.long)

    print(f"SELFIES: {selfies_str}")
    print(f"Encoded length: {len(encoded_selfies)}")

    # Set up properties
    property_token = 100  # Target property to predict
    available_properties = [10, 20, 30, 40, 50, 60]  # Other properties to use as context
    available_values = [0, 1]  # Binary values

    # Generate samples
    selfies_batch, properties, values, mask = generate_encoder_samples(
        selfies=selfies_tensor,
        nprops=4,  # Use 4 properties per sample (3 context + 1 target)
        property_token=property_token,
        available_properties=available_properties,
        available_values=available_values,
        num_samples=5,
        seed=42
    )

    print(f"\nGenerated batch shapes:")
    print(f"  selfies_batch: {selfies_batch.shape}")
    print(f"  properties:    {properties.shape}")
    print(f"  values:        {values.shape}")
    print(f"  mask:          {mask.shape}")

    print(f"\nFirst sample:")
    print(f"  Properties: {properties[0].tolist()}")
    print(f"  Values:     {values[0].tolist()}")
    print(f"  Mask:       {mask[0].tolist()}")
    print(f"  Target property (last position): {properties[0, -1].item()}")


def example_from_molecule():
    """Example using convenience wrapper with SELFIES string."""
    print("\n" + "=" * 80)
    print("Example 2: Using convenience wrapper with SELFIES string")
    print("=" * 80)

    tokenizer = spt.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

    selfies_str = "[C][C][C][C][O]"  # Butanol
    property_token = 200

    # This automatically handles encoding and default property/value selection
    selfies_batch, properties, values, mask = generate_encoder_samples_from_molecule(
        selfies_str=selfies_str,
        nprops=5,
        property_token=property_token,
        tokenizer=tokenizer,
        num_samples=10,
        seed=123,
        device='cpu'
    )

    print(f"SELFIES: {selfies_str}")
    print(f"\nGenerated batch shapes:")
    print(f"  selfies_batch: {selfies_batch.shape}")
    print(f"  properties:    {properties.shape}")
    print(f"  values:        {values.shape}")
    print(f"  mask:          {mask.shape}")


def example_property_permutations():
    """Example using property permutations for systematic exploration."""
    print("\n" + "=" * 80)
    print("Example 3: Generate all property orderings (permutations)")
    print("=" * 80)

    tokenizer = spt.SelfiesPropertyValTokenizer.load('brick/selfies_property_val_tokenizer')

    selfies_str = "[C][C][O]"  # Ethanol
    encoded_selfies = tokenizer.selfies_tokenizer.encode_selfies(selfies_str)
    selfies_tensor = torch.tensor(encoded_selfies, dtype=torch.long)

    property_token = 300  # Target property
    other_properties = [10, 20, 30]  # 3 context properties
    other_values = [1, 0, 1]  # Their corresponding values

    selfies_batch, properties, values, mask = generate_property_permutations(
        selfies=selfies_tensor,
        property_token=property_token,
        other_properties=other_properties,
        other_values=other_values
    )

    print(f"SELFIES: {selfies_str}")
    print(f"Context properties: {other_properties}")
    print(f"Context values:     {other_values}")
    print(f"Target property:    {property_token}")

    print(f"\nGenerated {selfies_batch.shape[0]} samples (3! = 6 permutations)")
    print(f"\nAll property orderings:")
    for i in range(selfies_batch.shape[0]):
        valid_props = properties[i][mask[i]].tolist()
        valid_vals = values[i][mask[i]].tolist()
        print(f"  Sample {i+1}: props={valid_props[:-1]} + [{valid_props[-1]}], vals={valid_vals}")


def example_with_model_inference():
    """Complete example with model inference (requires trained model)."""
    print("\n" + "=" * 80)
    print("Example 4: Complete workflow with model inference")
    print("=" * 80)

    try:
        import cvae.models.multitask_encoder as mte

        # Load model and tokenizer
        model_path = "cache/train_multitask_transformer_parallel/logs/split_0/models/me_roundrobin_property_dropout_V3/final_checkpoint"
        model = mte.MultitaskEncoder.load(model_path).eval()
        tokenizer = model.tokenizer

        print(f"Loaded model from: {model_path}")

        # Generate samples for inference
        selfies_str = "[C][C][C][O]"
        property_token = 100  # Adjust based on your model

        selfies_batch, properties, values, mask = generate_encoder_samples_from_molecule(
            selfies_str=selfies_str,
            nprops=5,
            property_token=property_token,
            tokenizer=tokenizer,
            num_samples=10,
            device='cpu'
        )

        print(f"\nRunning inference on {selfies_batch.shape[0]} samples...")

        # Run model inference
        with torch.no_grad():
            logits = model(selfies_batch, properties, values, mask)  # [B, nprops, num_classes]

            # Get predictions for the target property (at position nprops-1)
            target_logits = logits[:, -1, :]  # [B, num_classes]
            target_probs = torch.softmax(target_logits, dim=-1)
            prob_of_1 = target_probs[:, 1]  # Probability of class 1

        print(f"\nLogits shape: {logits.shape}")
        print(f"Target probabilities (class 1): {prob_of_1.tolist()}")
        print(f"Mean probability: {prob_of_1.mean().item():.4f}")
        print(f"Std deviation:    {prob_of_1.std().item():.4f}")

    except Exception as e:
        print(f"Could not run model inference: {e}")
        print("This example requires a trained model checkpoint.")


if __name__ == "__main__":
    # Run all examples
    example_basic_usage()
    example_from_molecule()
    example_property_permutations()
    example_with_model_inference()

    print("\n" + "=" * 80)
    print("All examples completed!")
    print("=" * 80)
