# GPTMed API Testing Guide

This guide walks you through testing the GPTMed package using the high-level API.

## Prerequisites

Install the gptmed package from PyPI:

```bash
pip install gptmed
```

Or install from local source in editable mode:

```bash
pip install -e ../gptmed
```

## Quick Start - Test the Complete Workflow

### Step 1: Generate a Configuration File

Create a training configuration file:

```bash
python run_gptmed.py create-config --output my_config.yaml
```

This creates a default YAML configuration file that you can customize.

### Step 2: Edit the Configuration

Open `my_config.yaml` and update the data paths to point to your tokenized data files:

```yaml
data:
  train_data: train.npy
  val_data: val.npy
```

The `train.npy` and `val.npy` files are already included in this directory.

### Step 3: Train the Model

Start training with your configuration:

```bash
python run_gptmed.py train --config my_config.yaml
```

This will:
- Load the configuration
- Initialize the model based on the specified size
- Train on your data
- Save checkpoints to `./checkpoints/`
- Log metrics to `./logs/`

**Note**: Training on CPU will be slow. For faster training, use a GPU-enabled machine.

### Step 4: Test Answer Generation

Once training is complete, test the model's ability to generate answers:

```bash
python run_gptmed.py generate \
    --checkpoint checkpoints/best_model.pt \
    --tokenizer medquad_tokenizer.model \
    --prompt "What is diabetes?" \
    --device cpu
```

You can customize generation parameters:

```bash
python run_gptmed.py generate \
    --checkpoint checkpoints/best_model.pt \
    --tokenizer medquad_tokenizer.model \
    --prompt "What are the symptoms of high blood pressure?" \
    --max-length 200 \
    --temperature 0.8 \
    --top-k 50 \
    --top-p 0.9 \
    --device cpu
```

### Step 5: Interactive Mode (Optional)

For a conversational experience:

```bash
python run_gptmed.py interactive \
    --checkpoint checkpoints/best_model.pt \
    --tokenizer medquad_tokenizer.model \
    --device cpu
```

Type your questions and get answers in real-time. Type `quit` or `exit` to stop.

## Configuration Options

The configuration file supports these main sections:

### Model Settings
```yaml
model:
  size: small  # Options: tiny, small, medium
```

### Data Paths
```yaml
data:
  train_data: train.npy
  val_data: val.npy
```

### Training Parameters
```yaml
training:
  num_epochs: 10
  batch_size: 16
  learning_rate: 0.0003
  weight_decay: 0.01
  grad_clip: 1.0
  warmup_steps: 100
```

### Device Settings
```yaml
device:
  device: cuda  # or 'cpu'
  seed: 42
```

### Checkpointing
```yaml
checkpointing:
  checkpoint_dir: ./checkpoints
  save_interval: 1
  keep_last_n: 3
```

## Understanding Training Results

After training, check the logs to evaluate performance:

```bash
# View training metrics
tail -20 logs/gpt_training_metrics.jsonl

# Check validation loss
grep "val_loss" logs/gpt_training_metrics.jsonl
```

**Good performance indicators:**
- Train loss: < 2.0 (ideally < 1.0)
- Validation loss: < 3.0 (ideally < 2.0)
- Perplexity: < 50

**If results are poor:**
- Train for more epochs (increase `num_epochs`)
- Use more training data
- Try a larger model size (`medium`)
- Adjust learning rate

## Files in This Directory

- `run_gptmed.py` - Main script for testing the API
- `train.npy` - Tokenized training data (6,473 samples)
- `val.npy` - Tokenized validation data (720 samples)
- `medquad_tokenizer.model` - Pre-trained tokenizer
- `example_config.yaml` - Example configuration file
- `checkpoints/` - Saved model checkpoints (created during training)
- `logs/` - Training logs and metrics (created during training)

## Troubleshooting

### Import Error: No module named 'gptmed'
```bash
pip install gptmed
```

### Import Error: No module named 'gptmed.services'
The package version is outdated. Upgrade to v0.3.5 or later:
```bash
pip install --upgrade gptmed
```

### Training is very slow
This is expected on CPU. Consider:
- Reducing batch size (e.g., `batch_size: 8`)
- Using a smaller model (`size: tiny`)
- Training on a GPU-enabled machine

### Poor generation quality
- Let training complete all epochs
- Check if validation loss is decreasing
- Ensure you have sufficient training data
- Try different generation parameters (temperature, top_k, top_p)

## Next Steps

After testing the API:
1. Prepare your own medical Q&A dataset
2. Tokenize it using the gptmed tokenizer
3. Train a larger model with more data
4. Fine-tune generation parameters for your use case
5. Deploy the model for production use

For more information, see the main [gptmed documentation](../gptmed/README.md).
