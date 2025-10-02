import torch
import pathlib
import sys

# Add the path to your modules if needed
# sys.path.append('/path/to/your/cvae/modules')

# Import your modules
import cvae.models.mixture_experts as me

def test_zero_loss_with_perfect_predictions():
    """
    Test that we get 0 loss when predictions perfectly match targets
    using the actual model's stratified loss function.
    """
    
    model = me.MoE.load(pathlib.Path("cache/finetune_benchmarks/models/moe"))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    
    # Get the stratified loss function
    lossfn = model.build_stratified_lossfn()
    tokenizer = model.tokenizer
    
    # Get value token information
    value_indexes = tokenizer.value_indexes()
    print(f"Value token mapping: {value_indexes}")
    
    value_token_ids = list(value_indexes.values())
    vocab_size = tokenizer.vocab_size
    
    # Create test data
    batch_size = 4
    seq_len = 20
    
    # Create output tensor with specific pattern:
    # - Position 0, 1: non-value tokens (ignored)
    # - Positions 2, 4, 6, 8, ... (even positions >= 2): value tokens
    # - Positions 3, 5, 7, 9, ... (odd positions >= 3): non-value tokens
    
    output = torch.zeros(batch_size, seq_len, dtype=torch.long, device=device)
    
    # Fill with random non-value tokens initially
    non_value_range = list(range(tokenizer.vocab_size))
    for vt in value_token_ids:
        if vt in non_value_range:
            non_value_range.remove(vt)
    
    for i in range(seq_len):
        if i < 2:  # Positions 0, 1
            output[:, i] = torch.randint(0, len(non_value_range), (batch_size,))
        elif i % 2 == 0:  # Even positions >= 2 (value positions)
            # Mix of both value tokens for testing
            output[:, i] = torch.randint(0, 2, (batch_size,)) * (value_token_ids[1] - value_token_ids[0]) + value_token_ids[0]
        else:  # Odd positions >= 3 (non-value positions)
            output[:, i] = torch.randint(0, len(non_value_range), (batch_size,))
    
    print(f"Output shape: {output.shape}")
    print(f"Sample output[0]: {output[0]}")
    
    # Create perfect predictions that match the output exactly
    # pred should be [batch_size, vocab_size, seq_len]
    pred = torch.zeros(batch_size, vocab_size, seq_len, device=device)
    
    # For each position, set the probability to 1.0 for the correct token
    for b in range(batch_size):
        for t in range(seq_len):
            correct_token = output[b, t].item()
            pred[b, correct_token, t] = 100.0  # High logit for correct token
            # Set small negative values for other tokens
            pred[b, :, t] = pred[b, :, t] - 10.0
            pred[b, correct_token, t] = 100.0  # Restore high value for correct token
    
    print(f"Prediction shape: {pred.shape}")
    
    # Test the loss function
    print("\nTesting loss function...")
    
    # Apply the loss function
    loss = lossfn(model.parameters(), pred, output)
    
    print(f"Loss with perfect predictions: {loss.item():.8f}")
    
    # Test with completely wrong predictions
    print("\nTesting with wrong predictions...")
    wrong_pred = pred.clone()
    
    # For value positions, flip the predictions
    for b in range(batch_size):
        for t in range(seq_len):
            if t >= 2 and t % 2 == 0:  # Value positions
                correct_token = output[b, t].item()
                # Find the other value token
                wrong_token = value_token_ids[1] if correct_token == value_token_ids[0] else value_token_ids[0]
                
                wrong_pred[b, correct_token, t] = -100.0  # Low logit for correct
                wrong_pred[b, wrong_token, t] = 100.0     # High logit for wrong
    
    wrong_loss = lossfn(model.parameters(), wrong_pred, output)
    print(f"Loss with wrong predictions: {wrong_loss.item():.8f}")
    
    # Analyze what the loss function is actually looking at
    print("\nAnalyzing loss function behavior...")
    
    # Check which positions are being considered
    ignore_index = tokenizer.pad_idx
    token_indices = torch.arange(seq_len, device=device)
    is_value_position = (token_indices >= 2) & (token_indices % 2 == 0)
    is_not_pad = (output != ignore_index)
    final_mask = is_value_position.unsqueeze(0).expand(batch_size, seq_len) & is_not_pad
    
    print(f"Value positions mask: {is_value_position}")
    print(f"Final mask for batch 0: {final_mask[0]}")
    print(f"Tokens at value positions for batch 0: {output[0][final_mask[0]]}")
    
    # Check if those tokens are actually value tokens
    value_tokens_tensor = torch.tensor(value_token_ids, device=device)
    actual_value_tokens = output[final_mask]
    is_actual_value = torch.isin(actual_value_tokens, value_tokens_tensor)
    print(f"Are they value tokens? {is_actual_value}")
    print(f"Number of actual value tokens found: {is_actual_value.sum().item()}")
    
    # Success criteria
    if loss.item() < 0.01:  # Very small loss indicates near-perfect predictions
        print("✅ Test PASSED: Perfect predictions yield very low loss")
        return True
    else:
        print("❌ Test FAILED: Perfect predictions should yield lower loss")
        return False

if __name__ == "__main__":
    test_zero_loss_with_perfect_predictions()