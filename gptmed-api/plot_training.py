#!/usr/bin/env python3
"""
Generate training metrics plot from JSONL file.

Usage:
    python plot_training.py                           # Use default file
    python plot_training.py logs/my_metrics.jsonl     # Specify file
"""

import json
import sys
from pathlib import Path

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("Error: matplotlib not installed. Run: pip install matplotlib")
    sys.exit(1)


def load_metrics(jsonl_path: str):
    """Load metrics from JSONL file."""
    steps, train_losses, moving_avg_losses = [], [], []
    val_steps, val_losses = [], []
    lr_steps, lr_values = [], []
    grad_steps, grad_norms = [], []
    ppl_steps, train_ppls = [], []
    val_ppl_steps, val_ppls = [], []
    
    with open(jsonl_path, 'r') as f:
        for line in f:
            entry = json.loads(line.strip())
            
            if entry.get('type') == 'step':
                step = entry['step']
                steps.append(step)
                train_losses.append(entry['loss'])
                if 'moving_avg_loss' in entry:
                    moving_avg_losses.append((step, entry['moving_avg_loss']))
                if 'learning_rate' in entry:
                    lr_steps.append(step)
                    lr_values.append(entry['learning_rate'])
                if 'grad_norm' in entry:
                    grad_steps.append(step)
                    grad_norms.append(entry['grad_norm'])
                if 'perplexity' in entry:
                    ppl_steps.append(step)
                    train_ppls.append(entry['perplexity'])
                    
            elif entry.get('type') == 'validation':
                val_steps.append(entry['step'])
                val_losses.append(entry['val_loss'])
                if 'val_perplexity' in entry:
                    val_ppl_steps.append(entry['step'])
                    val_ppls.append(entry['val_perplexity'])
    
    return {
        'steps': steps,
        'train_losses': train_losses,
        'moving_avg': moving_avg_losses,
        'val_steps': val_steps,
        'val_losses': val_losses,
        'lr_steps': lr_steps,
        'lr_values': lr_values,
        'grad_steps': grad_steps,
        'grad_norms': grad_norms,
        'ppl_steps': ppl_steps,
        'train_ppls': train_ppls,
        'val_ppl_steps': val_ppl_steps,
        'val_ppls': val_ppls,
    }


def plot_metrics(metrics: dict, output_path: str, title: str = "Training Metrics"):
    """Generate 4-panel training metrics plot."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # === Plot 1: Loss Curves ===
    ax1 = axes[0, 0]
    ax1.plot(metrics['steps'], metrics['train_losses'], alpha=0.3, label='Train Loss (raw)', color='blue')
    
    if metrics['moving_avg']:
        ma_steps, ma_values = zip(*metrics['moving_avg'])
        ax1.plot(ma_steps, ma_values, label='Train Loss (MA-100)', color='blue', linewidth=2)
    
    if metrics['val_steps']:
        ax1.plot(metrics['val_steps'], metrics['val_losses'], 'o-', label='Val Loss', color='orange', markersize=4)
    
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # === Plot 2: Learning Rate ===
    ax2 = axes[0, 1]
    if metrics['lr_steps']:
        ax2.plot(metrics['lr_steps'], metrics['lr_values'], color='green')
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Learning Rate')
    ax2.set_title('Learning Rate Schedule')
    ax2.grid(True, alpha=0.3)
    
    # === Plot 3: Gradient Norms ===
    ax3 = axes[1, 0]
    if metrics['grad_steps']:
        ax3.plot(metrics['grad_steps'], metrics['grad_norms'], alpha=0.5, color='red')
    ax3.set_xlabel('Step')
    ax3.set_ylabel('Gradient Norm')
    ax3.set_title('Gradient Norms')
    ax3.grid(True, alpha=0.3)
    
    # === Plot 4: Perplexity ===
    ax4 = axes[1, 1]
    if metrics['ppl_steps']:
        # Cap perplexity for visualization
        ppls = [min(p, 1000) for p in metrics['train_ppls']]
        ax4.plot(metrics['ppl_steps'], ppls, alpha=0.5, label='Train Perplexity', color='purple')
    if metrics['val_ppl_steps']:
        val_ppls = [min(p, 1000) for p in metrics['val_ppls']]
        ax4.plot(metrics['val_ppl_steps'], val_ppls, 'o-', label='Val Perplexity', color='magenta', markersize=4)
    ax4.set_xlabel('Step')
    ax4.set_ylabel('Perplexity')
    ax4.set_title('Perplexity')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"✅ Plot saved to: {output_path}")


def main():
    # Default paths
    default_jsonl = "logs/gptmed_training_metrics.jsonl"
    
    # Get input file
    jsonl_path = sys.argv[1] if len(sys.argv) > 1 else default_jsonl
    
    if not Path(jsonl_path).exists():
        print(f"Error: File not found: {jsonl_path}")
        sys.exit(1)
    
    print(f"📊 Loading metrics from: {jsonl_path}")
    metrics = load_metrics(jsonl_path)
    
    print(f"   Steps: {len(metrics['steps'])}")
    print(f"   Validations: {len(metrics['val_steps'])}")
    
    # Generate output filename
    stem = Path(jsonl_path).stem
    output_path = f"logs/{stem}_plot.png"
    
    # Get title from filename
    title = f"Training Metrics: {stem}"
    
    plot_metrics(metrics, output_path, title)


if __name__ == "__main__":
    main()
