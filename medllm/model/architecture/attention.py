"""
Multi-Head Causal Self-Attention

PURPOSE:
This is the core mechanism that allows the model to attend to different parts
of the input sequence. "Causal" means the model can only look at previous tokens,
not future ones (essential for next-token prediction).

WHAT THIS STEP DOES:
1. Linear projections: Create Query, Key, Value matrices
   - Input: [batch_size, seq_len, d_model]
   - Q, K, V each: [batch_size, seq_len, d_model]

2. Split into multiple heads
   - Reshape to: [batch_size, n_heads, seq_len, d_head]
   - where d_head = d_model / n_heads

3. Scaled dot-product attention
   - Compute attention scores: Q @ K^T / sqrt(d_head)
   - Apply causal mask (CRITICAL: prevents looking at future)
   - Softmax to get attention weights
   - Apply to values: attention_weights @ V

4. Concatenate heads and project back
   - Output: [batch_size, seq_len, d_model]

PACKAGES USED:
- torch: PyTorch tensors and operations
- torch.nn: Linear layers, Dropout
- torch.nn.functional: Softmax
- math: sqrt for scaling

FILES FROM THIS PROJECT:
- None (this is a base component)

TENSOR SHAPES EXPLAINED:
- n_heads: Number of attention heads (4-8)
- d_head: Dimension per head (d_model / n_heads)
- Causal mask: Lower triangular matrix [seq_len, seq_len]

COMMON FAILURE MODES TO AVOID:
- Missing causal mask → model cheats by seeing future tokens
- Wrong mask shape → silent failures or crashes
- Not scaling attention scores → vanishing/exploding gradients
- Forgetting dropout → overfitting
- Wrong tensor transpose/reshape → incorrect attention patterns
- Not masking padding tokens → attending to meaningless tokens
"""
