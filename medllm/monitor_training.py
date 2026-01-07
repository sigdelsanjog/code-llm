"""
Training Progress Monitor

PURPOSE:
Monitor training progress in real-time by reading metrics from log file.
Helps you understand if training is working and when to stop.

USAGE:
    python monitor_training.py                      # Monitor latest training
    python monitor_training.py --tail 20           # Show last 20 steps
    python monitor_training.py --watch             # Live monitoring

WHAT THIS SHOWS:
- Training loss over time
- Learning rate schedule
- Gradient norms (detect explosions)
- Validation loss (detect overfitting)
- Training speed (tokens/sec)
"""

import json
import argparse
from pathlib import Path
import time
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))


def load_metrics(log_file: Path):
    """Load metrics from JSONL file."""
    metrics = []
    
    if not log_file.exists():
        return metrics
    
    with open(log_file, 'r') as f:
        for line in f:
            if line.strip():
                metrics.append(json.loads(line))
    
    return metrics


def print_summary(metrics):
    """Print training summary statistics."""
    if not metrics:
        print("No metrics found yet.")
        return
    
    # Get latest metrics
    latest = metrics[-1]
    
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    
    print(f"\nCurrent Step: {latest.get('step', 'N/A')}")
    print(f"Time Elapsed: {latest.get('timestamp', 0):.1f}s")
    
    # Training metrics
    if 'train_loss' in latest:
        print(f"\nLatest Training Loss: {latest['train_loss']:.4f}")
    
    if 'learning_rate' in latest:
        print(f"Learning Rate: {latest['learning_rate']:.2e}")
    
    if 'grad_norm' in latest:
        print(f"Gradient Norm: {latest['grad_norm']:.3f}")
    
    if 'tokens_per_sec' in latest:
        print(f"Speed: {latest['tokens_per_sec']:.0f} tokens/sec")
    
    # Validation metrics
    val_metrics = [m for m in metrics if 'val_loss' in m]
    if val_metrics:
        latest_val = val_metrics[-1]
        print(f"\nLatest Validation Loss: {latest_val['val_loss']:.4f}")
        if 'val_perplexity' in latest_val:
            print(f"Validation Perplexity: {latest_val['val_perplexity']:.2f}")
    
    # Loss trends
    recent_losses = [m['train_loss'] for m in metrics[-100:] if 'train_loss' in m]
    if len(recent_losses) >= 10:
        avg_loss = sum(recent_losses) / len(recent_losses)
        min_loss = min(recent_losses)
        max_loss = max(recent_losses)
        
        print(f"\nLast 100 steps:")
        print(f"  Average Loss: {avg_loss:.4f}")
        print(f"  Min Loss: {min_loss:.4f}")
        print(f"  Max Loss: {max_loss:.4f}")


def print_recent_steps(metrics, n=20):
    """Print recent training steps."""
    print("\n" + "=" * 60)
    print(f"Last {n} Training Steps")
    print("=" * 60)
    print(f"\n{'Step':>6} | {'Loss':>8} | {'LR':>10} | {'Grad':>8} | {'Tok/s':>8}")
    print("-" * 60)
    
    recent = metrics[-n:]
    for m in recent:
        if 'train_loss' not in m:
            continue
        
        step = m.get('step', 0)
        loss = m.get('train_loss', 0)
        lr = m.get('learning_rate', 0)
        grad = m.get('grad_norm', 0)
        speed = m.get('tokens_per_sec', 0)
        
        print(f"{step:6d} | {loss:8.4f} | {lr:10.2e} | {grad:8.3f} | {speed:8.0f}")


def watch_training(log_file: Path, interval: int = 5):
    """Watch training progress in real-time."""
    print("Watching training (Ctrl+C to stop)...")
    print(f"Log file: {log_file}")
    print(f"Update interval: {interval}s\n")
    
    last_step = 0
    
    try:
        while True:
            metrics = load_metrics(log_file)
            
            if not metrics:
                print("Waiting for training to start...")
                time.sleep(interval)
                continue
            
            current_step = metrics[-1].get('step', 0)
            
            if current_step > last_step:
                # Clear screen (optional)
                # print("\033[2J\033[H")
                
                print_summary(metrics)
                print_recent_steps(metrics, n=10)
                
                last_step = current_step
            
            time.sleep(interval)
    
    except KeyboardInterrupt:
        print("\n\nStopped monitoring.")


def check_for_issues(metrics):
    """Check for common training issues."""
    print("\n" + "=" * 60)
    print("Issue Detection")
    print("=" * 60)
    
    if not metrics:
        print("No metrics to check.")
        return
    
    issues_found = False
    
    # Check for NaN loss
    nan_losses = [m for m in metrics if 'train_loss' in m and m['train_loss'] != m['train_loss']]
    if nan_losses:
        print("\n⚠️  WARNING: NaN loss detected!")
        print("   This usually means gradient explosion.")
        print("   Try: Lower learning rate, enable gradient clipping")
        issues_found = True
    
    # Check for high gradient norms
    high_grads = [m for m in metrics if 'grad_norm' in m and m['grad_norm'] > 10]
    if len(high_grads) > 10:
        print("\n⚠️  WARNING: Frequent high gradient norms!")
        print(f"   {len(high_grads)} steps with grad_norm > 10")
        print("   Try: Lower learning rate, stronger gradient clipping")
        issues_found = True
    
    # Check for stagnant loss
    recent_losses = [m['train_loss'] for m in metrics[-200:] if 'train_loss' in m]
    if len(recent_losses) >= 100:
        first_half_avg = sum(recent_losses[:50]) / 50
        second_half_avg = sum(recent_losses[-50:]) / 50
        improvement = first_half_avg - second_half_avg
        
        if improvement < 0.01:
            print("\n⚠️  WARNING: Loss not decreasing!")
            print(f"   Improvement in last 100 steps: {improvement:.4f}")
            print("   Try: Increase learning rate, check data quality")
            issues_found = True
    
    # Check overfitting
    val_metrics = [m for m in metrics if 'val_loss' in m and 'train_loss' in m]
    if len(val_metrics) >= 2:
        latest_val = val_metrics[-1]
        train_loss = latest_val.get('train_loss', 0)
        val_loss = latest_val.get('val_loss', 0)
        gap = val_loss - train_loss
        
        if gap > 0.5:
            print("\n⚠️  WARNING: Possible overfitting!")
            print(f"   Train loss: {train_loss:.4f}")
            print(f"   Val loss: {val_loss:.4f}")
            print(f"   Gap: {gap:.4f}")
            print("   Try: Add dropout, reduce model size, more data")
            issues_found = True
    
    if not issues_found:
        print("\n✅ No issues detected. Training looks healthy!")


def main():
    parser = argparse.ArgumentParser(description='Monitor training progress')
    
    parser.add_argument('--log-file', type=str, default='./logs/gpt_training_metrics.jsonl',
                        help='Path to metrics log file')
    parser.add_argument('--tail', type=int, default=20,
                        help='Number of recent steps to show')
    parser.add_argument('--watch', action='store_true',
                        help='Watch training in real-time')
    parser.add_argument('--interval', type=int, default=5,
                        help='Update interval for watch mode (seconds)')
    parser.add_argument('--check-issues', action='store_true',
                        help='Check for common training issues')
    
    args = parser.parse_args()
    
    log_file = Path(args.log_file)
    
    if args.watch:
        watch_training(log_file, args.interval)
    else:
        metrics = load_metrics(log_file)
        print_summary(metrics)
        print_recent_steps(metrics, n=args.tail)
        
        if args.check_issues:
            check_for_issues(metrics)


if __name__ == '__main__':
    main()
