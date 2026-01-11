"""
Complete API Workflow Test

This script demonstrates the complete workflow:
1. Create config
2. Train model  
3. Generate text

Run this to verify the entire GPTMed API works end-to-end.
"""

import gptmed
import os
from pathlib import Path

print("="*60)
print("GPTMed Complete API Workflow Test")
print("="*60)
print(f"GPTMed version: {gptmed.__version__}\n")

# Configuration
config_file = "workflow_test_config.yaml"
checkpoint_dir = "./model/checkpoints"
checkpoint_path = f"{checkpoint_dir}/best_model.pt"
tokenizer_path = "../medllm/tokenizer/medquad_tokenizer.model"

# Step 1: Create configuration
print("Step 1: Creating configuration...")
gptmed.create_config(config_file)
print(f"✅ Config created: {config_file}\n")

# Check if we should train or skip to generation
if os.path.exists(checkpoint_path):
    print(f"ℹ️  Checkpoint exists: {checkpoint_path}")
    print("Skipping training, going straight to generation.\n")
    skip_training = True
else:
    print("No checkpoint found. Will train the model.\n")
    skip_training = False

# Step 2: Train (if needed)
if not skip_training:
    print("Step 2: Training model...")
    print("Note: Using example_config.yaml with correct data paths\n")
    
    try:
        results = gptmed.train_from_config(
            config_path="example_config.yaml",
            verbose=True
        )
        
        print(f"\n✅ Training complete!")
        print(f"   Best checkpoint: {results['best_checkpoint']}")
        print(f"   Best val loss: {results['final_val_loss']:.4f}\n")
        
    except Exception as e:
        print(f"\n❌ Training failed: {e}")
        exit(1)
else:
    print("Step 2: Skipped (using existing checkpoint)\n")

# Step 3: Generate text
print("Step 3: Testing generation...")

test_prompts = [
    "What is diabetes?",
    "What causes high blood pressure?"
]

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n{'-'*60}")
    print(f"Prompt {i}: {prompt}")
    print('-'*60)
    
    try:
        answer = gptmed.generate(
            checkpoint=checkpoint_path,
            tokenizer=tokenizer_path,
            prompt=prompt,
            max_length=100,
            temperature=0.7,
            device="cuda"
        )
        
        print(f"Generated: {answer}")
        
    except Exception as e:
        print(f"❌ Generation failed: {e}")

print(f"\n{'='*60}")
print("✅ Complete workflow test finished!")
print('='*60)
