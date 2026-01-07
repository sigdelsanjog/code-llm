MASTER PROMPT (Phase 1 – From Scratch LLM)

I am building a causal language model (GPT-style) completely from scratch in PyTorch for learning purposes.

Constraints & context:

Dataset: MedQuAD (medical QA, open-source)

Goal: Phase 1 learning — tokenizer → objective → transformer → training

Hardware: NVIDIA GTX 1080 (8 GB VRAM), 12-core CPU, 16 GB RAM

I am NOT using Hugging Face transformers or pre-trained models

I want explicit control over tokenizer, attention, loss, and decoding

Current phase:
I am focusing only on:

MedQuAD data formatting for causal LM

Training a SentencePiece tokenizer

Defining the training objective and loss

Please assume:

I want minimal abstractions

I want code that is debuggable and educational

I care more about correctness than speed

Guide me step-by-step, explain why each design decision is made, and warn me about common failure modes (e.g., repetition collapse, bad tokenization, data leakage).

MASTER PROMPT — PHASE 2
Transformer Architecture (From Scratch)

I am in Phase 2 of building a causal Transformer (GPT-style) from scratch in PyTorch.

What is already done:

Dataset is cleaned and converted into plain causal text

SentencePiece tokenizer is trained (BPE, fixed vocab)

Training objective is next-token prediction with CrossEntropyLoss

Current focus:

Implementing the Transformer architecture manually

Components: token embeddings, positional embeddings, multi-head self-attention with causal masking, feed-forward layers, residual connections, layer normalization

Constraints:

No Hugging Face transformers

No pre-trained weights

Must fit on a single GTX 1080 GPU

Code must be minimal, readable, and debuggable

Expectations:

Explain tensor shapes at each step

Explain why causal masking is implemented the way it is

Highlight common architectural mistakes that cause training collapse or repetition

Provide a reference implementation suitable for learning, not production

Guide me step-by-step and challenge incorrect assumptions if I make any.

MASTER PROMPT — PHASE 3
Training Loop, Stability & Debugging

I am in Phase 3 of training a Transformer language model from scratch.

What exists:

Working GPT-style Transformer implementation

SentencePiece tokenizer

MedQuAD-based causal text dataset

Current focus:

Writing a full training loop in PyTorch

Batching, padding, attention masks

Learning rate scheduling (warmup + decay)

Gradient clipping

Constraints:

Single GPU (GTX 1080)

Small-to-medium batch sizes

No distributed training yet

Expectations:

Show how to detect silent training failures

Explain loss curves and what abnormal patterns indicate

Explain why repetition happens even when loss decreases

Help me build confidence that the model is actually learning

Treat this as an engineering debugging session, not a tutorial.

MASTER PROMPT — PHASE 4
Decoding, Sampling & Degeneration Control

I am in Phase 4: inference and text generation from a causal language model trained from scratch.

What works:

Model trains without crashing

Loss decreases

Current focus:

Implementing decoding strategies: greedy, temperature sampling, top-k, top-p

Understanding and fixing repetition loops

Logits scaling and numerical stability

Expectations:

Explain why greedy decoding fails

Show how sampling parameters interact with model entropy

Help me diagnose whether repetition is a model issue or decoding issue

I want principled explanations, not heuristic recipes.

MASTER PROMPT — PHASE 5
Scaling, Efficiency & Hardware Awareness

I am in Phase 5: scaling and optimizing my Transformer model.

Current status:

Single-GPU training works

Model architecture and training loop are stable

Current focus:

Model size vs VRAM trade-offs

Mixed precision training

Gradient accumulation

Efficient attention variants (conceptual understanding, not blind adoption)

Constraints:

GTX 1080

No multi-node training yet

Expectations:

Explain why certain optimizations help or hurt

Warn against premature optimization

Help me reason about FLOPs, memory, and throughput

Treat this as preparation for future distributed training.

MASTER PROMPT — PHASE 6
Domain Adaptation (Medical NLP)

I am in Phase 6: specializing a general language model for the medical domain.

What exists:

A working causal language model

Experience training from scratch

Current focus:

Domain adaptation using medical corpora (MedQuAD, PubMed, later MIMIC)

Avoiding catastrophic forgetting

Evaluating domain-specific language quality

Expectations:

Explain curriculum learning for domain adaptation

Explain why medical LMs fail silently

Help me design meaningful evaluation beyond loss

Assume I care about scientific correctness and safety.

MASTER PROMPT — PHASE 7
Distributed Training & “Real LLM” Thinking

I am in Phase 7: understanding how large language models are trained at scale.

Current focus:

Data parallelism vs model parallelism

Pipeline parallelism

Why modern LLM training is an infrastructure problem

Expectations:

Explain concepts without assuming massive clusters

Connect theory to real systems (SLURM, NCCL, FSDP, ZeRO)

Help me build intuition before touching large infrastructure

I want systems-level understanding, not surface-level buzzwords.

Final advice (important)

Most people never reach Phase 3.
You already avoided the biggest trap: import-first thinking.

If you document these prompts and actually follow them in order, you’ll end up with something rare:

The ability to reason about LLMs instead of just using them.

When you’re ready, come back with:

your MedQuAD text sample
or

your tokenizer config

We’ll continue from there.
