# GPTMed API Quick Start Examples

This file contains practical examples for using the gptmed-api runner.

## ✅ Successfully Verified

The PyPI package **gptmed v0.3.1** is now live with the high-level API!

Installation:

```bash
pip install gptmed
```

## Available Commands

### 1. Create Configuration File

```bash
python run_gptmed.py create-config --output my_config.yaml
```

Output:

```
✅ Configuration file created: my_config.yaml

📝 Next steps:
1. Edit my_config.yaml with your training settings
2. Run: python run_gptmed.py train --config my_config.yaml
```

### 2. Train a Model

```bash
# Edit your config file first, then:
python run_gptmed.py train --config my_config.yaml
```

This will:

- Load your configuration
- Create the model
- Train on your data
- Save checkpoints automatically
- Display training progress

### 3. Generate Text

```bash
python run_gptmed.py generate \
    --checkpoint ../gptmed/model/checkpoints/best_model.pt \
    --tokenizer ../medllm/tokenizer/medquad_tokenizer.model \
    --prompt "What is diabetes?" \
    --max-length 150 \
    --temperature 0.7
```

### 4. Interactive Mode

```bash
python run_gptmed.py interactive \
    --checkpoint ../gptmed/model/checkpoints/best_model.pt \
    --tokenizer ../medllm/tokenizer/medquad_tokenizer.model
```

Then you can ask multiple questions:

```
🤔 Enter your prompt: What is diabetes?
💭 Generating...
📝 Generated:
------------------------------------------------------------
Diabetes is a chronic condition that affects how your body...
------------------------------------------------------------

🤔 Enter your prompt: What causes it?
...
```

## Complete Workflow Example

```bash
# Step 1: Create config
python run_gptmed.py create-config --output medical_qa.yaml

# Step 2: Edit medical_qa.yaml
# - Set your data paths
# - Choose model size (tiny, small, medium)
# - Adjust hyperparameters

# Step 3: Train
python run_gptmed.py train --config medical_qa.yaml

# Step 4: Test generation
python run_gptmed.py generate \
    --checkpoint model/checkpoints/best_model.pt \
    --tokenizer tokenizer/medquad_tokenizer.model \
    --prompt "What is hypertension?"

# Step 5: Use interactively
python run_gptmed.py interactive \
    --checkpoint model/checkpoints/best_model.pt \
    --tokenizer tokenizer/medquad_tokenizer.model
```

## Python API Usage

You can also use the API directly in Python:

```python
import gptmed

# Create config
gptmed.create_config('my_config.yaml')

# Edit my_config.yaml, then train
results = gptmed.train_from_config('my_config.yaml')
print(f"Best model saved at: {results['best_checkpoint']}")

# Generate text
answer = gptmed.generate(
    checkpoint=results['best_checkpoint'],
    tokenizer='tokenizer/medquad_tokenizer.model',
    prompt='What is machine learning?',
    max_length=150,
    temperature=0.7
)
print(answer)
```

## Configuration File Example

```yaml
model:
  size: small # Options: tiny, small, medium

data:
  train_data: ./data/tokenized/train.npy
  val_data: ./data/tokenized/val.npy

training:
  num_epochs: 10
  batch_size: 16
  learning_rate: 0.0003
  weight_decay: 0.01
  grad_clip: 1.0
  warmup_steps: 100

optimizer:
  betas: [0.9, 0.95]
  eps: 1.0e-08

checkpointing:
  checkpoint_dir: ./model/checkpoints
  save_every: 1
  keep_last_n: 3

logging:
  log_dir: ./logs
  eval_every: 100
  log_every: 10

device:
  device: cuda # or cpu
  seed: 42

advanced:
  max_steps: -1 # -1 for full training
```

## Tips

1. **Model Size Selection**:

   - `tiny`: Fast training, good for testing (~1M params)
   - `small`: Balanced performance (~10M params)
   - `medium`: Better quality, slower (~25M params)

2. **Temperature Settings**:

   - Lower (0.3-0.5): More focused, deterministic
   - Medium (0.7-0.9): Balanced creativity
   - Higher (1.0-1.5): More creative, diverse

3. **Batch Size**:
   - Adjust based on your GPU memory
   - Smaller batch = less memory, slower training
   - Larger batch = more memory, faster training

## Verification Test

Run this to verify everything works:

```python
import gptmed

# Check version
print(f"GPTMed version: {gptmed.__version__}")

# Check API availability
print(f"API functions: {gptmed.__all__}")

# Verify functions exist
assert hasattr(gptmed, 'create_config')
assert hasattr(gptmed, 'train_from_config')
assert hasattr(gptmed, 'generate')

print("✅ All API functions available!")
```

## Next Steps

1. Prepare your training data (tokenized .npy files)
2. Create and edit a configuration file
3. Train your model
4. Generate responses
5. Iterate and improve

For more details, see the main gptmed documentation at https://pypi.org/project/gptmed/
