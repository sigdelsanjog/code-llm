"""
Compare Generation Strategies

PURPOSE:
Test the same prompt with different sampling strategies to see how they affect output quality.
Helps you understand:
- Greedy vs temperature vs top-k vs top-p
- Temperature effects on creativity/coherence
- Repetition control effectiveness

USAGE:
    python test_generation.py
    python test_generation.py --custom-prompt "Q: What is asthma?"
"""

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

from inference.generator import TextGenerator
from inference.generation_config import (
    GenerationConfig,
    get_greedy_config,
    get_balanced_config,
    get_creative_config,
    get_conservative_config
)


def test_strategies(generator, prompt, max_length=100):
    """Test all generation strategies on the same prompt."""
    
    strategies = {
        'Greedy (temp=0)': get_greedy_config(),
        'Conservative (temp=0.5)': get_conservative_config(),
        'Balanced (temp=0.7)': get_balanced_config(),
        'Creative (temp=1.0)': get_creative_config()
    }
    
    print("=" * 80)
    print(f"Prompt: {prompt}")
    print("=" * 80)
    
    for name, config in strategies.items():
        config.max_length = max_length
        
        print(f"\n{'='*80}")
        print(f"Strategy: {name}")
        print(f"Config: temp={config.temperature}, top_k={config.top_k}, "
              f"top_p={config.top_p}, rep_penalty={config.repetition_penalty}")
        print("=" * 80)
        
        output = generator.generate(
            prompt=prompt,
            gen_config=config,
            verbose=False
        )
        
        print(f"\n{output}\n")


def evaluate_quality(generator, test_prompts, config):
    """
    Generate for multiple prompts and check quality metrics.
    
    Quality indicators:
    - Unique tokens (diversity)
    - Repetition rate
    - Average token length
    """
    print("\n" + "=" * 80)
    print("Quality Evaluation")
    print("=" * 80)
    
    all_outputs = []
    
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n[{i}/{len(test_prompts)}] {prompt}")
        
        output = generator.generate(
            prompt=prompt,
            gen_config=config,
            verbose=False
        )
        
        all_outputs.append(output)
        
        # Basic quality metrics
        tokens = output.split()
        unique_tokens = len(set(tokens))
        total_tokens = len(tokens)
        diversity = unique_tokens / total_tokens if total_tokens > 0 else 0
        
        print(f"  Length: {total_tokens} tokens")
        print(f"  Unique: {unique_tokens} ({diversity*100:.1f}% diversity)")
        print(f"  Output: {output[:150]}...")
    
    return all_outputs


def main():
    parser = argparse.ArgumentParser(description='Test generation strategies')
    
    parser.add_argument('--checkpoint', type=str, default='./model/checkpoints/best_model.pt',
                        help='Model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='./tokenizer/medquad_tokenizer.model',
                        help='Tokenizer path')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'])
    parser.add_argument('--max-length', type=int, default=100,
                        help='Max generation length')
    parser.add_argument('--custom-prompt', type=str, default=None,
                        help='Custom prompt to test')
    parser.add_argument('--compare-strategies', action='store_true',
                        help='Compare all sampling strategies')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("Generation Strategy Comparison")
    print("=" * 80)
    
    # Load model
    print(f"\nLoading model...")
    generator = TextGenerator.from_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        tokenizer_path=Path(args.tokenizer),
        device=args.device
    )
    
    # Test prompts
    test_prompts = [
        "Q: What is diabetes?",
        "Q: What are the symptoms of flu?",
        "Q: How to prevent heart disease?"
    ]
    
    if args.custom_prompt:
        test_prompts = [args.custom_prompt]
    
    if args.compare_strategies:
        # Compare all strategies on one prompt
        for prompt in test_prompts[:1]:  # Just first prompt
            test_strategies(generator, prompt, args.max_length)
    else:
        # Evaluate quality with balanced config
        config = get_balanced_config()
        config.max_length = args.max_length
        
        outputs = evaluate_quality(generator, test_prompts, config)
    
    print("\n" + "=" * 80)
    print("Observations:")
    print("=" * 80)
    print("""
The model has learned:
✓ Medical vocabulary (diabetes, symptoms, treatment)
✓ Q&A format structure
✓ Some disease-related patterns

Issues to note:
✗ Repetitive phrases (memorization)
✗ Nonsensical combinations
✗ Formatting artifacts from training data

Why this happens:
- Small dataset (47K examples)
- Small model (~10M parameters)
- Limited training (2000 steps = ~5 epochs)

To improve:
1. More training data
2. Larger model
3. Longer training
4. Better preprocessing (remove formatting artifacts)
5. Fine-tuning on specific tasks
    """)


if __name__ == '__main__':
    main()
