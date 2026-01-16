# Explainable AI (XAI) for GptMed Transformer

> A comprehensive guide to understanding and interpreting your tiny language model's training process and behavior.

---

## Table of Contents

1. [Overview](#overview)
2. [Training Process Observability](#training-process-observability)
   - [Loss Curves & Metrics](#1-loss-curves--metrics)
   - [Gradient Flow Analysis](#2-gradient-flow-analysis)
   - [Weight Distribution Over Time](#3-weight-distribution-over-time)
   - [Learning Rate Analysis](#4-learning-rate-analysis)
3. [Model Behavior Interpretation](#model-behavior-interpretation)
   - [Attention Visualization](#5-attention-visualization)
   - [Saliency Maps / Input Attribution](#6-saliency-maps--input-attribution)
   - [Logit Lens / Layer Probing](#7-logit-lens--layer-probing)
   - [Embedding Space Analysis](#8-embedding-space-analysis)
4. [Tools & Libraries](#tools--libraries)
5. [Implementation Roadmap](#implementation-roadmap)
6. [Code Templates](#code-templates)
7. [Key Questions XAI Answers](#key-questions-xai-answers)

---

## Overview

### Why XAI Matters for Your Project

| Aspect      | Without XAI                 | With XAI                                                    |
| ----------- | --------------------------- | ----------------------------------------------------------- |
| Training    | "Loss went down"            | "Layer 3 gradients are vanishing, attention is diffuse"     |
| Debugging   | "Model gives wrong answers" | "Model attends to wrong tokens, embedding space is chaotic" |
| Improvement | Trial and error             | Data-driven decisions                                       |
| Trust       | Black box                   | Interpretable reasoning                                     |

### Two Categories of XAI

```
┌─────────────────────────────────────────────────────────────────────┐
│                    TRAINING PROCESS                                 │
│                    (What's happening during learning?)              │
├─────────────────────────────────────────────────────────────────────┤
│  • Loss Curves & Metrics                                           │
│  • Gradient Flow Analysis                                          │
│  • Weight Distribution Over Time                                   │
│  • Learning Rate vs Loss                                           │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                    MODEL BEHAVIOR                                   │
│                    (How is the model making decisions?)             │
├─────────────────────────────────────────────────────────────────────┤
│  • Attention Visualization                                         │
│  • Saliency Maps / Input Attribution                               │
│  • Logit Lens / Probing                                            │
│  • Embedding Space Analysis                                        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Training Process Observability

### 1. Loss Curves & Metrics

**Purpose**: Track how well the model is learning over time.

**What to Track**:

- Training loss (per step and per epoch)
- Validation loss (to detect overfitting)
- Perplexity (exponential of loss - more interpretable)
- Learning rate schedule

**Current Implementation**:
You already log to `logs/gpt_training_metrics.jsonl`

**Enhanced Metrics to Add**:

```python
metrics = {
    "step": step,
    "train_loss": loss.item(),
    "val_loss": val_loss,
    "perplexity": math.exp(loss.item()),
    "learning_rate": scheduler.get_last_lr()[0],
    "epoch": epoch,
    "tokens_seen": tokens_seen,
}
```

**What to Look For**:
| Pattern | Meaning | Action |
|---------|---------|--------|
| Train loss ↓, Val loss ↓ | Healthy learning | Continue |
| Train loss ↓, Val loss ↑ | Overfitting | Add dropout, reduce epochs, more data |
| Both loss plateau | Stuck | Increase LR, check data quality |
| Loss spikes | Instability | Reduce LR, increase warmup |
| Loss = NaN | Exploding gradients | Gradient clipping, lower LR |

**Visualization Tool**: TensorBoard, Weights & Biases

---

### 2. Gradient Flow Analysis

**Purpose**: Understand if gradients are propagating properly through all layers.

**Problems to Detect**:

- **Vanishing Gradients**: Deep layers don't learn (gradients → 0)
- **Exploding Gradients**: Training unstable (gradients → ∞)
- **Dead Neurons**: ReLU neurons that never activate

**Implementation**:

```python
def log_gradient_stats(model, step):
    """Log gradient statistics per layer."""
    gradient_stats = {}

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad
            gradient_stats[name] = {
                "mean": grad.mean().item(),
                "std": grad.std().item(),
                "max": grad.max().item(),
                "min": grad.min().item(),
                "norm": grad.norm(2).item(),
                "zero_fraction": (grad == 0).float().mean().item(),
            }

    return gradient_stats

def compute_total_gradient_norm(model):
    """Compute L2 norm of all gradients."""
    total_norm = 0.0
    for param in model.parameters():
        if param.grad is not None:
            total_norm += param.grad.data.norm(2).item() ** 2
    return total_norm ** 0.5
```

**What to Look For**:
| Metric | Healthy Range | Problem Indicator |
|--------|---------------|-------------------|
| Gradient norm | 0.1 - 10.0 | < 0.001 (vanishing) or > 100 (exploding) |
| Zero fraction | < 0.1 | > 0.5 (dead neurons) |
| Layer gradient ratio | Similar across layers | 100x difference between layers |

**Visualization**: Gradient flow plot showing norm per layer

```
Layer 1: ████████████████ (norm: 1.2)
Layer 2: ███████████████  (norm: 1.1)
Layer 3: ██████████       (norm: 0.7)  ← Potential issue
Layer 4: ███              (norm: 0.2)  ← Vanishing
```

---

### 3. Weight Distribution Over Time

**Purpose**: Track how model weights evolve during training.

**What to Track**:

- Weight magnitude per layer
- Weight change (delta) per step
- Weight initialization vs final weights
- Attention weight patterns

**Implementation**:

```python
def log_weight_stats(model, step):
    """Log weight statistics per layer."""
    weight_stats = {}

    for name, param in model.named_parameters():
        weight_stats[name] = {
            "mean": param.data.mean().item(),
            "std": param.data.std().item(),
            "max": param.data.max().item(),
            "min": param.data.min().item(),
            "norm": param.data.norm(2).item(),
        }

    return weight_stats

def track_weight_changes(model, previous_weights):
    """Track how much weights changed since last checkpoint."""
    changes = {}
    for name, param in model.named_parameters():
        if name in previous_weights:
            delta = (param.data - previous_weights[name]).norm(2).item()
            changes[name] = delta
    return changes
```

**What to Look For**:
| Pattern | Meaning |
|---------|---------|
| Weights barely change | Layer not learning (vanishing gradients) |
| Weights change wildly | Unstable training |
| One layer dominates | Imbalanced learning |
| Weights → 0 or → ∞ | Initialization or LR problem |

---

### 4. Learning Rate Analysis

**Purpose**: Understand relationship between learning rate and training dynamics.

**Techniques**:

#### LR Range Test (Smith, 2017)

```python
def lr_range_test(model, train_loader, min_lr=1e-7, max_lr=10, num_steps=100):
    """
    Find optimal learning rate by gradually increasing LR.
    Plot loss vs LR to find the steepest descent point.
    """
    lrs = []
    losses = []

    lr = min_lr
    lr_multiplier = (max_lr / min_lr) ** (1 / num_steps)

    for step, batch in enumerate(train_loader):
        if step >= num_steps:
            break

        # Set LR
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # Forward + backward
        loss = compute_loss(model, batch)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        lrs.append(lr)
        losses.append(loss.item())
        lr *= lr_multiplier

    return lrs, losses
    # Optimal LR is where loss decreases fastest (steepest slope)
```

**What to Look For**:

```
Loss
  │
  │\
  │ \
  │  \___      ← Optimal LR range
  │      \___
  │          \___/  ← LR too high, loss increases
  └─────────────────── Learning Rate
     1e-5   1e-3   1e-1
```

---

## Model Behavior Interpretation

### 5. Attention Visualization

**Purpose**: See what tokens the model focuses on when making predictions.

**Why It Matters**: Transformers use self-attention to determine which tokens influence each other. Visualizing this reveals:

- Does the model attend to relevant medical terms?
- Is attention focused or scattered?
- Do different heads specialize?

**Requirements**:
Modify `GPTTransformer` to return attention weights.

**Implementation**:

#### Step 1: Modify Model to Return Attention

```python
# In gptmed/model/architecture.py

class MultiHeadAttention(nn.Module):
    def forward(self, x, mask=None, return_attention=False):
        # ... existing code ...

        # Compute attention weights
        attn_weights = F.softmax(scores / math.sqrt(self.head_dim), dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        attn_output = torch.matmul(attn_weights, v)

        if return_attention:
            return attn_output, attn_weights
        return attn_output

class GPTTransformer(nn.Module):
    def forward(self, x, mask=None, return_attention=False):
        attention_weights = []

        x = self.embedding(x)
        x = self.pos_encoding(x)

        for layer in self.layers:
            if return_attention:
                x, attn = layer(x, mask, return_attention=True)
                attention_weights.append(attn)
            else:
                x = layer(x, mask)

        x = self.ln_final(x)
        logits = self.output_projection(x)

        if return_attention:
            return logits, attention_weights
        return logits
```

#### Step 2: Visualize Attention

```python
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def visualize_attention(model, tokenizer, text, layer_idx=0, head_idx=0):
    """
    Visualize attention weights for a given input.

    Args:
        model: GPTTransformer model
        tokenizer: SentencePiece tokenizer
        text: Input text to analyze
        layer_idx: Which transformer layer to visualize
        head_idx: Which attention head to visualize
    """
    # Tokenize
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.id_to_piece(tid) for tid in token_ids]

    # Get attention weights
    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([token_ids])
        logits, attention_weights = model(input_tensor, return_attention=True)

    # Extract specific layer and head
    # attention_weights[layer] shape: (batch, num_heads, seq_len, seq_len)
    attn = attention_weights[layer_idx][0, head_idx].cpu().numpy()

    # Plot heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attn,
        xticklabels=tokens,
        yticklabels=tokens,
        cmap='Blues',
        annot=True if len(tokens) < 15 else False,
        fmt='.2f'
    )
    plt.xlabel('Key (attending to)')
    plt.ylabel('Query (attending from)')
    plt.title(f'Attention Weights - Layer {layer_idx}, Head {head_idx}')
    plt.tight_layout()
    plt.savefig(f'attention_layer{layer_idx}_head{head_idx}.png')
    plt.show()

def visualize_all_heads(model, tokenizer, text, layer_idx=0):
    """Visualize all attention heads for a layer."""
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.id_to_piece(tid) for tid in token_ids]

    model.eval()
    with torch.no_grad():
        input_tensor = torch.tensor([token_ids])
        logits, attention_weights = model(input_tensor, return_attention=True)

    attn = attention_weights[layer_idx][0].cpu().numpy()  # (num_heads, seq, seq)
    num_heads = attn.shape[0]

    fig, axes = plt.subplots(2, num_heads // 2, figsize=(4 * num_heads // 2, 8))
    axes = axes.flatten()

    for head_idx in range(num_heads):
        sns.heatmap(attn[head_idx], ax=axes[head_idx], cmap='Blues', cbar=False)
        axes[head_idx].set_title(f'Head {head_idx}')
        axes[head_idx].set_xticks([])
        axes[head_idx].set_yticks([])

    plt.suptitle(f'All Attention Heads - Layer {layer_idx}')
    plt.tight_layout()
    plt.savefig(f'attention_all_heads_layer{layer_idx}.png')
    plt.show()
```

**What to Look For**:

| Pattern            | Meaning                             | Example                          |
| ------------------ | ----------------------------------- | -------------------------------- |
| Diagonal attention | Token attends to itself             | Common in early layers           |
| Vertical stripes   | All tokens attend to specific token | Often punctuation or key terms   |
| Focused blocks     | Phrase-level attention              | Good for understanding structure |
| Uniform/diffuse    | No clear pattern                    | Model may be undertrained        |

**Medical Domain Test**:

```python
# Test with medical text
visualize_attention(model, tokenizer,
    "What causes type 2 diabetes mellitus?",
    layer_idx=2, head_idx=0
)

# Expected: Strong attention between "diabetes" and "causes"
# Or between "type 2" and "diabetes mellitus"
```

**Library Alternative**: Use `BertViz` for interactive visualization

```bash
pip install bertviz
```

---

### 6. Saliency Maps / Input Attribution

**Purpose**: Determine which input tokens most influenced the model's output.

**Techniques**:

| Technique            | Description                     | Pros                   | Cons          |
| -------------------- | ------------------------------- | ---------------------- | ------------- |
| Vanilla Gradient     | Gradient of output w.r.t. input | Fast, simple           | Noisy         |
| Integrated Gradients | Average gradients along path    | Theoretically grounded | Slower        |
| Attention Rollout    | Combine attention across layers | Uses existing weights  | Approximation |
| SHAP                 | Game-theoretic attribution      | Rigorous               | Very slow     |

**Implementation with Captum**:

```bash
pip install captum
```

```python
from captum.attr import IntegratedGradients, Saliency, LayerIntegratedGradients
import torch

class ModelWrapper(torch.nn.Module):
    """Wrapper to make model compatible with Captum."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_embeds):
        # Bypass embedding layer, use provided embeddings
        x = input_embeds
        x = self.model.pos_encoding(x)
        for layer in self.model.layers:
            x = layer(x)
        x = self.model.ln_final(x)
        logits = self.model.output_projection(x)
        return logits

def compute_saliency(model, tokenizer, text, target_position=-1):
    """
    Compute saliency scores showing which input tokens
    influenced the prediction at target_position.

    Args:
        model: GPTTransformer
        tokenizer: SentencePiece tokenizer
        text: Input text
        target_position: Which output position to explain (-1 = last)

    Returns:
        tokens: List of token strings
        saliency_scores: Attribution score per token
    """
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.id_to_piece(tid) for tid in token_ids]

    # Get embeddings
    input_tensor = torch.tensor([token_ids])
    embeddings = model.embedding(input_tensor)
    embeddings.requires_grad_(True)

    # Forward pass
    model.eval()

    # Create wrapper that takes embeddings directly
    wrapped_model = ModelWrapper(model)

    # Get output logits
    logits = wrapped_model(embeddings)

    # Get the predicted token at target position
    target_logit = logits[0, target_position, :].max()

    # Backward pass
    target_logit.backward()

    # Saliency = gradient magnitude
    saliency = embeddings.grad.abs().sum(dim=-1)[0]
    saliency_scores = saliency.detach().cpu().numpy()

    # Normalize
    saliency_scores = saliency_scores / saliency_scores.max()

    return tokens, saliency_scores

def visualize_saliency(tokens, saliency_scores):
    """Visualize saliency as colored text."""
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    fig, ax = plt.subplots(figsize=(12, 2))

    # Create colored boxes for each token
    for i, (token, score) in enumerate(zip(tokens, saliency_scores)):
        color = plt.cm.Reds(score)
        ax.add_patch(plt.Rectangle((i, 0), 1, 1, color=color))
        ax.text(i + 0.5, 0.5, token, ha='center', va='center', fontsize=10)

    ax.set_xlim(0, len(tokens))
    ax.set_ylim(0, 1)
    ax.axis('off')
    ax.set_title('Input Saliency (darker = more important)')
    plt.tight_layout()
    plt.savefig('saliency_map.png')
    plt.show()
```

**Using Captum's Integrated Gradients** (more accurate):

```python
from captum.attr import IntegratedGradients

def integrated_gradients_attribution(model, tokenizer, text, target_position=-1):
    """
    Use Integrated Gradients for more accurate attribution.
    """
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.id_to_piece(tid) for tid in token_ids]

    input_tensor = torch.tensor([token_ids])
    embeddings = model.embedding(input_tensor)

    wrapped_model = ModelWrapper(model)

    # Baseline = zero embeddings (or padding token embeddings)
    baseline = torch.zeros_like(embeddings)

    # Integrated Gradients
    ig = IntegratedGradients(wrapped_model)

    # Compute attributions for the predicted class at target position
    def forward_func(embeds):
        logits = wrapped_model(embeds)
        return logits[0, target_position, :].max(dim=-1)[0]

    attributions = ig.attribute(
        embeddings,
        baselines=baseline,
        n_steps=50,
        return_convergence_delta=False
    )

    # Sum across embedding dimension
    attr_scores = attributions.sum(dim=-1)[0].detach().cpu().numpy()
    attr_scores = np.abs(attr_scores)
    attr_scores = attr_scores / attr_scores.max()

    return tokens, attr_scores
```

**What to Look For**:

- High saliency on question words ("what", "why", "how")
- High saliency on domain-specific terms
- Low saliency on stopwords (good sign)
- Unexpected high saliency (potential bias)

---

### 7. Logit Lens / Layer Probing

**Purpose**: Understand what the model "knows" at each layer.

**Concept**: At each transformer layer, project the hidden state to vocabulary space and see what the model would predict if we stopped there.

**Implementation**:

```python
def logit_lens(model, tokenizer, text):
    """
    See how predictions evolve through layers.

    Returns predictions at each layer for the last token position.
    """
    token_ids = tokenizer.encode(text)
    tokens = [tokenizer.id_to_piece(tid) for tid in token_ids]

    model.eval()
    with torch.no_grad():
        # Get embeddings
        x = model.embedding(torch.tensor([token_ids]))
        x = model.pos_encoding(x)

        layer_predictions = []

        for layer_idx, layer in enumerate(model.layers):
            x = layer(x)

            # Project to vocabulary (using final layer norm and projection)
            normed = model.ln_final(x)
            logits = model.output_projection(normed)

            # Get top prediction at last position
            probs = F.softmax(logits[0, -1, :], dim=-1)
            top_prob, top_idx = probs.max(dim=-1)
            top_token = tokenizer.id_to_piece(top_idx.item())

            # Get top 5
            top5_probs, top5_indices = probs.topk(5)
            top5_tokens = [tokenizer.id_to_piece(idx.item()) for idx in top5_indices]

            layer_predictions.append({
                'layer': layer_idx,
                'top_token': top_token,
                'top_prob': top_prob.item(),
                'top5': list(zip(top5_tokens, top5_probs.tolist()))
            })

        return layer_predictions

def visualize_logit_lens(layer_predictions):
    """Visualize how predictions change through layers."""
    print("\nLogit Lens - Prediction Evolution Through Layers")
    print("=" * 60)

    for pred in layer_predictions:
        print(f"\nLayer {pred['layer']}:")
        print(f"  Top prediction: '{pred['top_token']}' ({pred['top_prob']:.3f})")
        print(f"  Top 5: {pred['top5']}")
```

**What to Look For**:
| Observation | Interpretation |
|-------------|----------------|
| Early layers predict syntax | Normal - learning structure first |
| Later layers predict semantics | Good - building understanding |
| Prediction stable across layers | Confident prediction |
| Prediction changes in last layer | Late refinement |
| Random predictions throughout | Model undertrained |

**Example Output**:

```
Input: "The patient has diabetes so they need"

Layer 0: 'the' (0.15)    ← Just echoing common tokens
Layer 1: 'to' (0.20)     ← Basic syntax
Layer 2: 'insulin' (0.25) ← Starting to understand context
Layer 3: 'insulin' (0.45) ← More confident
Layer 4: 'insulin' (0.62) ← Final prediction
```

---

### 8. Embedding Space Analysis

**Purpose**: Understand how the model organizes concepts internally.

**Questions to Answer**:

- Are similar medical terms clustered together?
- Are symptoms near their diseases?
- Are treatments near their conditions?

**Implementation**:

```python
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def extract_embeddings(model, tokenizer, words):
    """Extract embeddings for a list of words."""
    embeddings = []
    valid_words = []

    for word in words:
        # Tokenize and get first token
        token_ids = tokenizer.encode(word)
        if len(token_ids) > 0:
            # Get embedding for first token
            token_id = token_ids[0]
            emb = model.embedding.weight[token_id].detach().cpu().numpy()
            embeddings.append(emb)
            valid_words.append(word)

    return np.array(embeddings), valid_words

def visualize_embeddings_tsne(model, tokenizer, word_groups, output_path='embeddings_tsne.png'):
    """
    Visualize embeddings using t-SNE, colored by category.

    Args:
        model: GPTTransformer
        tokenizer: SentencePiece tokenizer
        word_groups: Dict of category -> list of words
            Example: {
                'diseases': ['diabetes', 'cancer', 'flu'],
                'symptoms': ['fever', 'pain', 'cough'],
                'treatments': ['insulin', 'surgery', 'antibiotics']
            }
    """
    all_words = []
    all_categories = []
    all_embeddings = []

    for category, words in word_groups.items():
        embeddings, valid_words = extract_embeddings(model, tokenizer, words)
        all_embeddings.append(embeddings)
        all_words.extend(valid_words)
        all_categories.extend([category] * len(valid_words))

    all_embeddings = np.vstack(all_embeddings)

    # Apply t-SNE
    tsne = TSNE(n_components=2, perplexity=min(30, len(all_words) - 1), random_state=42)
    coords = tsne.fit_transform(all_embeddings)

    # Plot
    plt.figure(figsize=(12, 8))

    categories = list(word_groups.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))

    for i, category in enumerate(categories):
        mask = np.array(all_categories) == category
        plt.scatter(coords[mask, 0], coords[mask, 1],
                   c=[colors[i]], label=category, s=100)

    # Add labels
    for i, word in enumerate(all_words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]),
                    fontsize=8, alpha=0.8)

    plt.legend()
    plt.title('Embedding Space Visualization (t-SNE)')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.tight_layout()
    plt.savefig(output_path)
    plt.show()

def compute_embedding_similarity(model, tokenizer, word_pairs):
    """
    Compute cosine similarity between word pairs.

    Args:
        word_pairs: List of (word1, word2) tuples

    Returns:
        List of (word1, word2, similarity) tuples
    """
    from torch.nn.functional import cosine_similarity

    results = []
    for word1, word2 in word_pairs:
        emb1, _ = extract_embeddings(model, tokenizer, [word1])
        emb2, _ = extract_embeddings(model, tokenizer, [word2])

        if len(emb1) > 0 and len(emb2) > 0:
            sim = cosine_similarity(
                torch.tensor(emb1),
                torch.tensor(emb2)
            ).item()
            results.append((word1, word2, sim))

    return results
```

**Medical Domain Test**:

```python
# Test medical concept organization
medical_words = {
    'diseases': ['diabetes', 'cancer', 'asthma', 'arthritis', 'hypertension'],
    'symptoms': ['fever', 'pain', 'cough', 'fatigue', 'nausea'],
    'treatments': ['insulin', 'chemotherapy', 'antibiotics', 'surgery', 'therapy'],
    'body_parts': ['heart', 'lung', 'brain', 'kidney', 'liver'],
}

visualize_embeddings_tsne(model, tokenizer, medical_words)

# Test relationships
pairs = [
    ('diabetes', 'insulin'),      # Should be similar (treatment)
    ('heart', 'cardiac'),         # Should be similar (synonym)
    ('fever', 'temperature'),     # Should be similar (related)
    ('diabetes', 'surgery'),      # Should be less similar
]

similarities = compute_embedding_similarity(model, tokenizer, pairs)
for w1, w2, sim in similarities:
    print(f"{w1} <-> {w2}: {sim:.3f}")
```

**What to Look For**:
| Pattern | Good Sign | Bad Sign |
|---------|-----------|----------|
| Disease clustering | Diseases grouped together | Diseases scattered randomly |
| Symptom-disease proximity | Related symptoms near diseases | No relationship |
| Antonym separation | "healthy" far from "sick" | Antonyms clustered |
| Hierarchical structure | "heart disease" near "heart" and "disease" | No compositionality |

---

## Tools & Libraries

### Recommended Stack

| Tool                 | Purpose                      | Installation                   | Difficulty |
| -------------------- | ---------------------------- | ------------------------------ | ---------- |
| **TensorBoard**      | Training metrics             | `pip install tensorboard`      | Easy       |
| **Weights & Biases** | Experiment tracking          | `pip install wandb`            | Easy       |
| **Captum**           | Attribution methods          | `pip install captum`           | Medium     |
| **BertViz**          | Attention visualization      | `pip install bertviz`          | Medium     |
| **Ecco**             | Transformer interpretability | `pip install ecco`             | Medium     |
| **TransformerLens**  | Mechanistic interpretability | `pip install transformer-lens` | Advanced   |

### TensorBoard Setup

```python
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('runs/gptmed_experiment')

# During training
writer.add_scalar('Loss/train', train_loss, step)
writer.add_scalar('Loss/val', val_loss, step)
writer.add_scalar('LearningRate', lr, step)
writer.add_histogram('Gradients/layer1', model.layers[0].weight.grad, step)

# Launch: tensorboard --logdir=runs
```

### Weights & Biases Setup

```python
import wandb

wandb.init(project="gptmed", config={
    "model_size": "tiny",
    "learning_rate": 1e-4,
    "epochs": 10,
})

# During training
wandb.log({
    "train_loss": train_loss,
    "val_loss": val_loss,
    "gradient_norm": grad_norm,
})
```

---

## Implementation Roadmap

### Phase 1: Training Observability (Week 1)

- [ ] Add gradient norm logging to trainer
- [ ] Set up TensorBoard or W&B
- [ ] Implement LR range test
- [ ] Track weight statistics per layer

### Phase 2: Attention Analysis (Week 2)

- [ ] Modify `GPTTransformer` to return attention weights
- [ ] Implement `visualize_attention()` function
- [ ] Create attention entropy metric
- [ ] Test with medical vs random text

### Phase 3: Input Attribution (Week 3)

- [ ] Install and configure Captum
- [ ] Implement saliency maps
- [ ] Implement integrated gradients
- [ ] Create visualization tools

### Phase 4: Embedding Analysis (Week 4)

- [ ] Extract learned embeddings
- [ ] Implement t-SNE visualization
- [ ] Test medical concept clustering
- [ ] Compute similarity matrices

### Phase 5: Integration (Week 5)

- [ ] Create unified XAI dashboard
- [ ] Add XAI endpoints to backend API
- [ ] Integrate visualizations into chat-ui
- [ ] Document findings

---

## Key Questions XAI Answers

| Question                                | Technique                   | Expected Insight                        |
| --------------------------------------- | --------------------------- | --------------------------------------- |
| Why did the model give this answer?     | Attention + Saliency        | Which input tokens drove the prediction |
| Is my model actually learning?          | Loss curves + Gradient flow | Training health and progress            |
| What does layer N understand?           | Logit lens                  | How understanding builds through layers |
| Are medical terms related in embedding? | t-SNE + Similarity          | Semantic organization quality           |
| Is attention focused or scattered?      | Attention entropy           | Model confidence and relevance          |
| Which heads specialize in what?         | Per-head attention analysis | Functional specialization               |
| Are gradients flowing to all layers?    | Gradient norm per layer     | Training stability                      |
| What would improve the model?           | All techniques combined     | Data, architecture, or training changes |

---

## Quick Reference Commands

```bash
# Install all XAI tools
pip install tensorboard wandb captum bertviz matplotlib seaborn scikit-learn

# Launch TensorBoard
tensorboard --logdir=logs

# Start W&B
wandb login
```

```python
# Quick attention visualization
from xai_tools import visualize_attention
visualize_attention(model, tokenizer, "What causes diabetes?")

# Quick saliency map
from xai_tools import compute_saliency, visualize_saliency
tokens, scores = compute_saliency(model, tokenizer, "What causes diabetes?")
visualize_saliency(tokens, scores)

# Quick embedding analysis
from xai_tools import visualize_embeddings_tsne
visualize_embeddings_tsne(model, tokenizer, medical_words)
```

---

## Resources

### Papers

- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Original transformer
- [A Survey on Explainable AI](https://arxiv.org/abs/2006.00093) - XAI overview
- [Integrated Gradients](https://arxiv.org/abs/1703.01365) - Attribution method

### Tutorials

- [Captum Tutorials](https://captum.ai/tutorials/)
- [BertViz Documentation](https://github.com/jessevig/bertviz)
- [TransformerLens Documentation](https://neelnanda-io.github.io/TransformerLens/)

### Blog Posts

- [The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/)
- [Visualizing Attention](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

---

_Last Updated: January 2026_
_Project: GptMed - Medical Language Model_
