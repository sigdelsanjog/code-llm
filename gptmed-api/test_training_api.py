"""
Test GPTMed API - Training

This script tests the gptmed API training function using PyPI-installed package.
"""

import gptmed

print("="*60)
print("Testing GPTMed Training API")
print("="*60)
print(f"\nGPTMed version: {gptmed.__version__}")
print(f"Available functions: {gptmed.__all__}\n")

# Step 1: Create a config file
print("Step 1: Creating configuration file...")
config_file = "training_test_config.yaml"
gptmed.create_config(config_file)
print(f"✅ Created: {config_file}\n")

# The config will have default paths, let's inform user to update them
print("📝 Note: Edit the config file to update data paths:")
print("   - train_data: ../medllm/data/tokenized/train.npy")
print("   - val_data: ../medllm/data/tokenized/val.npy")
print("\nOr use the example_config.yaml which already has correct paths.\n")

# Step 2: Train the model using example config
print("Step 2: Training model with example_config.yaml...")
print("Starting training...\n")

try:
    results = gptmed.train_from_config(
        config_path="example_config.yaml",
        verbose=True
    )
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60)
    print(f"Best checkpoint: {results['best_checkpoint']}")
    print(f"Best val loss: {results['final_val_loss']:.4f}")
    print(f"Total epochs: {results['total_epochs']}")
    
except Exception as e:
    print(f"\n❌ Error during training: {e}")
    import traceback
    traceback.print_exc()
