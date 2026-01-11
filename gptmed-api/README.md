# GPTMed API Runner

This folder contains an easy-to-use interface for training and using GPTMed models through the high-level `gptmed.api` module.

## Overview

The `run_gptmed.py` script provides a simple command-line interface that wraps the GPTMed API, making it easy to:

- Create training configuration files
- Train models with a single command
- Generate text from trained models
- Use models in interactive mode

## Installation

Install the gptmed package from PyPI:

```bash
pip install gptmed
```

This will install the latest version of gptmed with the high-level API.

## Quick Start

### Using the Pre-configured Example

We provide a ready-to-use configuration that points to the existing MedQuAD data:

```bash
# Train with the example configuration
python run_gptmed.py train --config example_config.yaml
```

The `example_config.yaml` already has the correct paths:

- Training data: `train.npy` (6,473 samples)
- Validation data: `val.npy` (720 samples)
- Tokenizer: `medquad_tokenizer.model`

**See [DATA_GUIDE.md](DATA_GUIDE.md) for details about the data files.**

## Usage

### 1. Create a Training Configuration

```bash
python run_gptmed.py create-config --output my_config.yaml
```

This creates a default YAML configuration file that you can edit with your training settings.

### 2. Edit the Configuration

Open `my_config.yaml` and customize:

- Model size (tiny, small, medium)
- **Training data paths** - Update to point to your tokenized .npy files:
  - For MedQuAD data: `../medllm/data/tokenized/train.npy` and `val.npy`
  - For custom data: See [DATA_GUIDE.md](DATA_GUIDE.md)
- Hyperparameters (learning rate, batch size, etc.)
- Output directories

**Example:**

```yaml
data:
  train_data: ../medllm/data/tokenized/train.npy
  val_data: ../medllm/data/tokenized/val.npy
```

### 3. Train a Model

```bash
python run_gptmed.py train --config my_config.yaml
```

This will:

- Load your configuration
- Create and initialize the model
- Train on your data
- Save checkpoints automatically
- Display training progress

### 4. Generate Text

Once trained, generate text using your model:

```bash
python run_gptmed.py generate \
    --checkpoint ../gptmed/model/checkpoints/best_model.pt \
    --tokenizer ../medllm/tokenizer/medquad_tokenizer.model \
    --prompt "What is diabetes?" \
    --max-length 150 \
    --temperature 0.7
```

### 5. Interactive Mode

For a more interactive experience:

```bash
python run_gptmed.py interactive \
    --checkpoint ../gptmed/model/checkpoints/best_model.pt \
    --tokenizer ../medllm/tokenizer/medquad_tokenizer.model \
    --max-length 150
```

This opens an interactive prompt where you can ask multiple questions without reloading the model each time.

## Command Reference

### create-config

Create a default training configuration file.

```bash
python run_gptmed.py create-config [--output PATH]
```

Options:

- `--output`, `-o`: Output path for config file (default: training_config.yaml)

### train

Train a model from a configuration file.

```bash
python run_gptmed.py train --config PATH
```

Options:

- `--config`, `-c`: Path to YAML configuration file (required)

### generate

Generate text using a trained model.

```bash
python run_gptmed.py generate \
    --checkpoint PATH \
    --tokenizer PATH \
    --prompt TEXT \
    [OPTIONS]
```

Required:

- `--checkpoint`: Path to model checkpoint (.pt file)
- `--tokenizer`: Path to tokenizer model (.model file)
- `--prompt`, `-p`: Input text/question

Optional:

- `--max-length`: Maximum tokens to generate (default: 100)
- `--temperature`: Sampling temperature, higher = more random (default: 0.7)
- `--top-k`: Top-k sampling parameter (default: 50)
- `--top-p`: Nucleus sampling parameter (default: 0.9)
- `--device`: Device to use: cuda or cpu (default: cuda)

### interactive

Interactive text generation mode.

```bash
python run_gptmed.py interactive \
    --checkpoint PATH \
    --tokenizer PATH \
    [OPTIONS]
```

Required:

- `--checkpoint`: Path to model checkpoint (.pt file)
- `--tokenizer`: Path to tokenizer model (.model file)

Optional:

- `--max-length`: Maximum tokens to generate (default: 100)
- `--temperature`: Sampling temperature (default: 0.7)
- `--top-k`: Top-k sampling (default: 50)
- `--top-p`: Nucleus sampling (default: 0.9)
- `--device`: Device to use: cuda or cpu (default: cuda)

## Examples

### Complete Workflow

```bash
# 1. Create configuration
python run_gptmed.py create-config --output medical_config.yaml

# 2. Edit medical_config.yaml with your settings
# (Set data paths, model size, hyperparameters, etc.)

# 3. Train the model
python run_gptmed.py train --config medical_config.yaml

# 4. Test generation
python run_gptmed.py generate \
    --checkpoint ../gptmed/model/checkpoints/best_model.pt \
    --tokenizer ../medllm/tokenizer/medquad_tokenizer.model \
    --prompt "What causes high blood pressure?"

# 5. Use interactively
python run_gptmed.py interactive \
    --checkpoint ../gptmed/model/checkpoints/best_model.pt \
    --tokenizer ../medllm/tokenizer/medquad_tokenizer.model
```

### Different Generation Settings

```bash
# More conservative generation (lower temperature)
python run_gptmed.py generate \
    --checkpoint model.pt \
    --tokenizer tokenizer.model \
    --prompt "Explain diabetes" \
    --temperature 0.5

# More creative generation (higher temperature)
python run_gptmed.py generate \
    --checkpoint model.pt \
    --tokenizer tokenizer.model \
    --prompt "Explain diabetes" \
    --temperature 1.0

# Longer response
python run_gptmed.py generate \
    --checkpoint model.pt \
    --tokenizer tokenizer.model \
    --prompt "Explain diabetes" \
    --max-length 300
```

## Benefits of Using This API

1. **Simplicity**: No need to write training loops or manage checkpoints manually
2. **Configuration-based**: All settings in one YAML file, easy to version control
3. **Flexible**: Supports different model sizes and hyperparameters
4. **Interactive**: Test your model quickly with interactive mode
5. **Production-ready**: Uses the same API as the main gptmed package

## Troubleshooting

### Module not found error

Make sure gptmed is installed from PyPI:

```bash
pip install gptmed
```

### CUDA out of memory

Try:

- Reducing batch size in config
- Using a smaller model size
- Using `--device cpu` for generation

### Checkpoint not found

Ensure the checkpoint path is correct. After training, the best model is saved at:

```
<checkpoint_dir>/best_model.pt
```

## Next Steps

- Visit [PyPI gptmed package](https://pypi.org/project/gptmed/) for more information
- Check the gptmed documentation for comprehensive guides
- Join the community for support and updates
