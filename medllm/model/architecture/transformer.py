"""
Full GPT-Style Transformer Model

PURPOSE:
Assembles all components (embeddings, decoder blocks, output projection)
into a complete causal language model for next-token prediction.

WHAT THIS STEP DOES:
1. Token + Positional Embeddings
   - Convert input IDs to vectors: [batch_size, seq_len, d_model]

2. Stack of N Decoder Blocks
   - Each block applies self-attention + FFN
   - Typically N = 4-6 layers for our hardware

3. Final Layer Normalization
   - Stabilize outputs before projection

4. Output Projection (Language Modeling Head)
   - Project to vocabulary: [batch_size, seq_len, vocab_size]
   - No activation (raw logits for CrossEntropyLoss)

5. Forward Pass
   - Input: token IDs [batch_size, seq_len]
   - Output: logits [batch_size, seq_len, vocab_size]

PACKAGES USED:
- torch: PyTorch tensors
- torch.nn: Module, Linear, LayerNorm, ModuleList

FILES FROM THIS PROJECT:
- architecture/embeddings.py: TokenPositionalEmbedding
- architecture/decoder_block.py: TransformerDecoderBlock
- configs/model_config.py: Hyperparameters (d_model, n_layers, etc.)

TENSOR SHAPES:
- Input IDs: [batch_size, seq_len] (integers)
- Embeddings: [batch_size, seq_len, d_model]
- After blocks: [batch_size, seq_len, d_model]
- Logits: [batch_size, seq_len, vocab_size]

HYPERPARAMETERS (for GTX 1080):
- vocab_size: 8000 (from tokenizer)
- d_model: 256-512 (embedding dimension)
- n_layers: 4-6 (number of transformer blocks)
- n_heads: 4-8 (attention heads)
- d_ff: 4 * d_model (FFN hidden size)
- dropout: 0.1-0.2
- max_seq_len: 512

COMMON FAILURE MODES TO AVOID:
- Not tying embeddings and output weights → slower convergence
- Too many layers → OOM or slow training
- d_model not divisible by n_heads → shape mismatch
- Missing final LayerNorm → unstable outputs
- Forgetting to handle padding mask → attending to padding
"""
