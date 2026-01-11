# Data Files Guide for GPTMed API

This guide explains where to find the data files needed for training and generation.

## Available Data Files

### Training Data (Tokenized)

The MedQuAD dataset has already been processed and tokenized in the `medllm` folder:

**Location**: `/medllm/data/tokenized/`

- **train.npy** - Training data (6,473 samples × 512 tokens)
- **val.npy** - Validation data (720 samples × 512 tokens)
- **metadata.json** - Dataset metadata

### Tokenizer Files

**Location**: `/medllm/tokenizer/`

- **medquad_tokenizer.model** - SentencePiece tokenizer trained on MedQuAD
- **medquad_tokenizer.vocab** - Vocabulary file

Also available at:

- `/backend/models/gptmed/gptmed_tokenizer.model`
- `/gptmed/gptmed/tokenizer/medquad_tokenizer.model`

## Configuration File Setup

When creating your config file, use these paths (relative to `gptmed-api` folder):

```yaml
data:
  train_data: ../medllm/data/tokenized/train.npy
  val_data: ../medllm/data/tokenized/val.npy
```

## Example Configuration

See `example_config.yaml` in this folder for a complete working configuration:

```bash
# View the example config
cat example_config.yaml

# Or create your own
python run_gptmed.py create-config --output my_config.yaml

# Then edit my_config.yaml and update the data paths:
# data:
#   train_data: ../medllm/data/tokenized/train.npy
#   val_data: ../medllm/data/tokenized/val.npy
```

## Data Information

### Dataset Statistics

```
Training samples: 6,473 Q&A pairs
Validation samples: 720 Q&A pairs
Sequence length: 512 tokens per sample
Source: MedQuAD (Medical Question Answering Dataset)
```

### Tokenizer Information

The tokenizer was trained on the MedQuAD dataset with:

- Vocabulary size: ~8,000 tokens
- Model type: SentencePiece BPE
- Special tokens: `<s>`, `</s>`, `<unk>`, `<pad>`

## Training with the Data

Once you have your config file with the correct paths:

```bash
# Using example config
python run_gptmed.py train --config example_config.yaml

# Or with your custom config
python run_gptmed.py train --config my_config.yaml
```

## Generation with Trained Model

After training, you'll have checkpoints in `./model/checkpoints/`. Use them for generation:

```bash
python run_gptmed.py generate \
    --checkpoint ./model/checkpoints/best_model.pt \
    --tokenizer ../medllm/tokenizer/medquad_tokenizer.model \
    --prompt "What is diabetes?"
```

## If You Want to Use Your Own Data

If you want to train on different data:

1. **Prepare your data** in Q&A format (one pair per line)
2. **Train a tokenizer** on your data
3. **Tokenize your data** to create .npy files
4. **Update config** to point to your new data files

See the medllm folder for examples of how the data was prepared:

- `/medllm/preprocess.py` - Data preprocessing
- `/medllm/tokenizer/train_tokenizer.py` - Tokenizer training

## Quick Verification

Check that the data files are accessible:

```bash
cd gptmed-api
python3 -c "
import numpy as np
import os

# Check data files exist
train_path = '../medllm/data/tokenized/train.npy'
val_path = '../medllm/data/tokenized/val.npy'
tokenizer_path = '../medllm/tokenizer/medquad_tokenizer.model'

print('✓ Train data exists:', os.path.exists(train_path))
print('✓ Val data exists:', os.path.exists(val_path))
print('✓ Tokenizer exists:', os.path.exists(tokenizer_path))

# Load and check shapes
train = np.load(train_path)
val = np.load(val_path)

print(f'\nTrain shape: {train.shape}')
print(f'Val shape: {val.shape}')
print('\n✅ All data files ready for training!')
"
```

## Summary

**For Training:**

- Data: `../medllm/data/tokenized/{train,val}.npy`
- Config: `example_config.yaml` (already set up)

**For Generation:**

- Checkpoint: `./model/checkpoints/best_model.pt` (after training)
- Tokenizer: `../medllm/tokenizer/medquad_tokenizer.model`

**Ready to train?**

```bash
python run_gptmed.py train --config example_config.yaml
```
