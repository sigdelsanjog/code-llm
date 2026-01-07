"""
Transformer Decoder Block

PURPOSE:
Combines attention, feed-forward network, residual connections, and layer
normalization into a single reusable block. Multiple blocks are stacked
to form the full transformer.

WHAT THIS STEP DOES:
1. Multi-head causal self-attention
   - With residual connection: x + attention(x)
   - With layer normalization

2. Feed-forward network
   - With residual connection: x + ffn(x)
   - With layer normalization

Architecture pattern (Pre-LN vs Post-LN):
We use Pre-LN (normalize before sublayer) because it's more stable:
  x = x + attention(LayerNorm(x))
  x = x + ffn(LayerNorm(x))

PACKAGES USED:
- torch: PyTorch tensors
- torch.nn: Module, LayerNorm, Dropout

FILES FROM THIS PROJECT:
- architecture/attention.py: Multi-head attention module
- architecture/feedforward.py: FFN module

TENSOR SHAPES:
- Input: [batch_size, seq_len, d_model]
- Output: [batch_size, seq_len, d_model] (unchanged)

COMMON FAILURE MODES TO AVOID:
- Post-LN instead of Pre-LN → training instability
- Forgetting residual connections → vanishing gradients
- Wrong LayerNorm dimension → incorrect normalization
- Dropout too high → underfitting
- Dropout too low → overfitting
"""
