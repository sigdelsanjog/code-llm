"""
Interactive Text Generation Script

PURPOSE:
Test your trained model by generating text from prompts.
Useful for checking model quality and experimenting with generation settings.

USAGE:
    # Generate from best model
    python generate_sample.py
    
    # Use specific checkpoint
    python generate_sample.py --checkpoint model/checkpoints/checkpoint_step_1000.pt
    
    # Try different sampling strategies
    python generate_sample.py --strategy creative

WHAT THIS DOES:
1. Load trained model
2. Load tokenizer
3. Generate text from medical prompts
4. Compare different generation strategies
"""

import argparse
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from inference.generator import TextGenerator
from inference.generation_config import (
    get_greedy_config,
    get_balanced_config,
    get_creative_config,
    get_conservative_config
)


def main():
    parser = argparse.ArgumentParser(description='Generate text with trained model')
    
    parser.add_argument('--checkpoint', type=str, default='./model/checkpoints/best_model.pt',
                        help='Path to model checkpoint')
    parser.add_argument('--tokenizer', type=str, default='./tokenizer/medquad_tokenizer.model',
                        help='Path to tokenizer')
    parser.add_argument('--strategy', type=str, default='balanced',
                        choices=['greedy', 'balanced', 'creative', 'conservative'],
                        help='Generation strategy')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to run on')
    parser.add_argument('--prompt', type=str, default=None,
                        help='Custom prompt (if not provided, uses test prompts)')
    parser.add_argument('--max-length', type=int, default=200,
                        help='Maximum generation length')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Medical LLM - Text Generation")
    print("=" * 60)
    
    # Load generator
    print(f"\nLoading model from: {args.checkpoint}")
    generator = TextGenerator.from_checkpoint(
        checkpoint_path=Path(args.checkpoint),
        tokenizer_path=Path(args.tokenizer),
        device=args.device
    )
    
    # Select generation config
    print(f"\nGeneration strategy: {args.strategy}")
    if args.strategy == 'greedy':
        gen_config = get_greedy_config()
    elif args.strategy == 'balanced':
        gen_config = get_balanced_config()
    elif args.strategy == 'creative':
        gen_config = get_creative_config()
    else:  # conservative
        gen_config = get_conservative_config()
    
    gen_config.max_length = args.max_length
    
    # Test prompts
    test_prompts = [
        "Q: What is diabetes?",
        "Q: What are the symptoms of COVID-19?",
        "Q: How is hypertension treated?",
        "Q: What causes cancer?",
        "Q: What is a vaccine?"
    ]
    
    if args.prompt:
        # Use custom prompt
        prompts = [args.prompt]
    else:
        # Use test prompts
        prompts = test_prompts
    
    # Generate
    print("\n" + "=" * 60)
    print("Generating Responses")
    print("=" * 60)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[{i}/{len(prompts)}] Prompt: {prompt}")
        print("-" * 60)
        
        output = generator.generate(
            prompt=prompt,
            gen_config=gen_config,
            verbose=False
        )
        
        print(f"Output:\n{output}")
        print("-" * 60)
    
    # Interactive mode
    print("\n" + "=" * 60)
    print("Interactive Mode (Ctrl+C to exit)")
    print("=" * 60)
    
    try:
        while True:
            prompt = input("\nEnter prompt: ").strip()
            
            if not prompt:
                continue
            
            output = generator.generate(
                prompt=prompt,
                gen_config=gen_config,
                verbose=False
            )
            
            print(f"\nGenerated:\n{output}\n")
    
    except KeyboardInterrupt:
        print("\n\nExiting...")


if __name__ == '__main__':
    main()
