"""
Token and Positional Embeddings

PURPOSE:
This file converts token IDs (integers) into continuous vector representations
that the Transformer can process. It also adds positional information so the
model knows the order of tokens (since attention has no inherent notion of position).

WHAT THIS STEP DOES:
1. Token Embedding: Maps each token ID to a learned vector of size d_model
   - Input: [batch_size, seq_len] of token IDs
   - Output: [batch_size, seq_len, d_model] of vectors

2. Positional Embedding: Adds position information to each token
   - Two approaches: learned embeddings or sinusoidal encoding
   - Same shape as token embeddings: [batch_size, seq_len, d_model]

3. Combines both: token_emb + pos_emb
   - Final output: [batch_size, seq_len, d_model]

PACKAGES USED:
- torch: PyTorch tensors and neural network modules
- torch.nn: Embedding layer, Dropout

FILES FROM THIS PROJECT:
- None (this is a base component)

TENSOR SHAPES EXPLAINED:
- batch_size: Number of sequences processed together
- seq_len: Length of each sequence (512 in our case)
- vocab_size: Size of tokenizer vocabulary (8000)
- d_model: Embedding dimension (256-512)

COMMON FAILURE MODES TO AVOID:
- Forgetting dropout → overfitting
- Wrong positional encoding dimension → shape mismatch
- Not scaling embeddings → training instability
- Using fixed positions > max_seq_len → index out of bounds
"""
