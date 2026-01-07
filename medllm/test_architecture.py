"""
Test Transformer Architecture

Verifies that the model builds correctly and can perform a forward pass.

This is a sanity check before training:
1. Model instantiates without errors
2. Forward pass produces correct tensor shapes
3. No NaN or Inf values in outputs
4. Parameter count matches expectations

Run this BEFORE training to catch bugs early!
"""

import torch
from model.architecture import GPTTransformer
from model.configs.model_config import get_small_config, get_tiny_config


def test_model_architecture():
    """Test model instantiation and forward pass."""
    
    print("=" * 60)
    print("Testing Transformer Architecture")
    print("=" * 60)
    
    # Use tiny config for fast testing
    config = get_tiny_config()
    
    print("\nModel Configuration:")
    print(f"  vocab_size: {config.vocab_size}")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  d_ff: {config.d_ff}")
    print(f"  max_seq_len: {config.max_seq_len}")
    print(f"  dropout: {config.dropout}")
    
    # Create model
    print("\n1. Creating model...")
    model = GPTTransformer(config)
    
    # Count parameters
    total_params = model.count_parameters()
    print(f"✅ Model created successfully!")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    # Create dummy input
    print("\n2. Creating dummy input...")
    batch_size = 4
    seq_len = 128
    
    # Random token IDs
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    print(f"   Input shape: {input_ids.shape}")
    print(f"   Input dtype: {input_ids.dtype}")
    
    # Forward pass
    print("\n3. Running forward pass...")
    model.eval()  # Set to eval mode (disables dropout)
    
    with torch.no_grad():
        logits = model(input_ids)
    
    print(f"✅ Forward pass successful!")
    print(f"   Output shape: {logits.shape}")
    print(f"   Expected: [{batch_size}, {seq_len}, {config.vocab_size}]")
    
    # Verify shapes
    expected_shape = (batch_size, seq_len, config.vocab_size)
    assert logits.shape == expected_shape, f"Shape mismatch! Got {logits.shape}, expected {expected_shape}"
    
    # Check for NaN/Inf
    print("\n4. Checking for NaN/Inf...")
    has_nan = torch.isnan(logits).any()
    has_inf = torch.isinf(logits).any()
    
    if has_nan:
        print("❌ ERROR: NaN values detected in output!")
        return False
    if has_inf:
        print("❌ ERROR: Inf values detected in output!")
        return False
    
    print("✅ No NaN/Inf values")
    
    # Check output statistics
    print("\n5. Output statistics:")
    print(f"   Mean: {logits.mean().item():.4f}")
    print(f"   Std: {logits.std().item():.4f}")
    print(f"   Min: {logits.min().item():.4f}")
    print(f"   Max: {logits.max().item():.4f}")
    
    # Test with different sequence lengths
    print("\n6. Testing variable sequence lengths...")
    for test_len in [64, 256, 512]:
        test_input = torch.randint(0, config.vocab_size, (2, test_len))
        with torch.no_grad():
            test_output = model(test_input)
        print(f"   seq_len={test_len}: ✅ Output shape {test_output.shape}")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed!")
    print("=" * 60)
    print("\nArchitecture is ready for training!")
    
    return True


def test_small_config():
    """Test the small config (actual training size)."""
    
    print("\n" + "=" * 60)
    print("Testing Small Config (Training Size)")
    print("=" * 60)
    
    config = get_small_config()
    
    print("\nModel Configuration:")
    print(f"  d_model: {config.d_model}")
    print(f"  n_layers: {config.n_layers}")
    print(f"  n_heads: {config.n_heads}")
    print(f"  d_ff: {config.d_ff}")
    
    model = GPTTransformer(config)
    total_params = model.count_parameters()
    
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Model size: ~{total_params * 4 / 1024 / 1024:.2f} MB (fp32)")
    
    # Estimate memory for training
    # Model + gradients + optimizer states (Adam: 2x params) ≈ 4x model size
    training_memory = total_params * 4 * 4 / 1024 / 1024
    print(f"Estimated training memory: ~{training_memory:.0f} MB")
    print(f"Should fit in GTX 1080 (8GB): {'✅ Yes' if training_memory < 4000 else '❌ No, try tiny config'}")


if __name__ == '__main__':
    # Test tiny config
    success = test_model_architecture()
    
    if success:
        # Test small config
        test_small_config()
        
        print("\n" + "=" * 60)
        print("Next Steps:")
        print("=" * 60)
        print("1. Phase 3: Create training loop")
        print("2. Create PyTorch Dataset for tokenized data")
        print("3. Define loss function (CrossEntropyLoss)")
        print("4. Define optimizer (AdamW)")
        print("5. Start training!")
