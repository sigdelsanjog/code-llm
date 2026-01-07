"""
Feed-Forward Network (FFN)

PURPOSE:
Applies position-wise transformations to each token independently.
This adds non-linear processing power to the model beyond what attention provides.

WHAT THIS STEP DOES:
1. First linear projection: Expand dimension
   - Input: [batch_size, seq_len, d_model]
   - Output: [batch_size, seq_len, d_ff]
   - Typically d_ff = 4 * d_model (expansion)

2. Non-linear activation (GELU or ReLU)
   - Introduces non-linearity
   - GELU is smoother than ReLU, often better for transformers

3. Second linear projection: Project back
   - Input: [batch_size, seq_len, d_ff]
   - Output: [batch_size, seq_len, d_model]

4. Dropout for regularization

PACKAGES USED:
- torch: PyTorch tensors
- torch.nn: Linear, Dropout, GELU/ReLU

FILES FROM THIS PROJECT:
- None (this is a base component)

TENSOR SHAPES EXPLAINED:
- d_ff: Hidden dimension in FFN (usually 4 * d_model)
- For d_model=256, d_ff=1024
- For d_model=512, d_ff=2048

COMMON FAILURE MODES TO AVOID:
- d_ff too small → insufficient expressiveness
- d_ff too large → OOM on GPU, overfitting
- Forgetting activation → just linear transformation (useless)
- Using ReLU with high learning rate → dead neurons
"""
