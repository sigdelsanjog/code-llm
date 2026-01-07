"""
Model Configuration

PURPOSE:
Central place to store all model hyperparameters. Makes it easy to experiment
with different model sizes without changing code.

WHAT THIS FILE CONTAINS:
1. ModelConfig dataclass with all hyperparameters:
   - vocab_size: From tokenizer (8000)
   - d_model: Embedding/hidden dimension
   - n_layers: Number of transformer blocks
   - n_heads: Number of attention heads
   - d_ff: Feed-forward hidden dimension
   - dropout: Dropout probability
   - max_seq_len: Maximum sequence length
   
2. Predefined configurations:
   - Tiny: For quick testing (d_model=128, n_layers=2)
   - Small: For GTX 1080 training (d_model=256, n_layers=4)
   - Medium: Larger if memory allows (d_model=512, n_layers=6)

PACKAGES USED:
- dataclasses: For clean config structure
- json: For saving/loading configs

FILES FROM THIS PROJECT:
- None (this defines configs for other files to use)

DESIGN DECISIONS:
- d_model must be divisible by n_heads
- d_ff typically 4 * d_model (expansion ratio)
- dropout 0.1-0.2 (too high → underfitting, too low → overfitting)
- max_seq_len matches tokenization (512)

MEMORY ESTIMATION (approximate):
- Model parameters ≈ 12 * n_layers * d_model^2
- Small config: ~10M parameters (~40MB)
- Medium config: ~40M parameters (~160MB)
- Fits comfortably in 8GB VRAM with batch_size=16-32
"""
