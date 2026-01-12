#!/usr/bin/env python3
"""
GPTMed API Runner

Simple command-line interface for training and using GPTMed models
using the high-level gptmed.api module.
"""

import argparse
import sys
import yaml
from pathlib import Path

try:
    import gptmed
except ImportError as e:
    print(f"Error: Could not import gptmed package. Please install it first:")
    print(f"  pip install gptmed")
    print(f"\nDetails: {e}")
    sys.exit(1)


def create_default_config():
    """Create a default configuration dictionary"""
    return {
        'model': {
            'size': 'small',  # Options: tiny, small, medium
        },
        'data': {
            'train_data': 'train.npy',
            'val_data': 'val.npy',
        },
        'training': {
            'num_epochs': 10,
            'batch_size': 16,
            'learning_rate': 0.0003,
            'weight_decay': 0.01,
            'grad_clip': 1.0,
            'warmup_steps': 100,
        },
        'optimizer': {
            'betas': [0.9, 0.95],
            'eps': 1.0e-08,
        },
        'checkpointing': {
            'checkpoint_dir': './model/checkpoints',
            'save_every': 1,
            'keep_last_n': 3,
        },
        'logging': {
            'log_dir': './logs',
            'eval_every': 100,
            'log_every': 10,
        },
        'device': {
            'device': 'cuda',  # Change to 'cpu' if no GPU
            'seed': 42,
        },
        'advanced': {
            'max_steps': -1,  # -1 for full training
        }
    }


def cmd_create_config(args):
    """Create a default configuration file"""
    output_path = Path(args.output)
    
    if output_path.exists() and not args.force:
        print(f"Error: {output_path} already exists. Use --force to overwrite.")
        return 1
    
    # Use gptmed API to create config
    gptmed.create_config(str(output_path))
    
    print(f"✓ Created configuration file: {output_path}")
    print(f"\nNext steps:")
    print(f"  1. Edit {output_path} to customize your training settings")
    print(f"  2. Update data paths to point to your tokenized .npy files")
    print(f"  3. Run training: python run_gptmed.py train --config {output_path}")
    
    return 0


def cmd_train(args):
    """Train a model using the configuration file"""
    config_path = Path(args.config)
    
    if not config_path.exists():
        print(f"Error: Configuration file not found: {config_path}")
        return 1
    
    print(f"Loading configuration from {config_path}...")
    
    print(f"\n{'='*60}")
    print(f"Starting Training with GPTMed API")
    print(f"{'='*60}\n")
    
    try:
        # Train using gptmed API
        print("Starting training...\n")
        gptmed.train_from_config(str(config_path))
        
        print(f"\n{'='*60}")
        print(f"Training Complete!")
        print(f"{'='*60}")
        
        return 0
        
    except Exception as e:
        print(f"\n{'='*60}")
        print(f"Error during training: {e}")
        print(f"{'='*60}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_generate(args):
    """Generate text using a trained model"""
    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1
    
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found: {tokenizer_path}")
        return 1
    
    print(f"Loading model from {checkpoint_path}...")
    print(f"Using tokenizer: {tokenizer_path}")
    
    try:
        print(f"\nPrompt: {args.prompt}")
        print(f"{'='*60}")
        
        # Generate using gptmed API
        response = gptmed.generate(
            checkpoint=str(checkpoint_path),
            tokenizer=str(tokenizer_path),
            prompt=args.prompt,
            max_length=args.max_length,
            temperature=args.temperature,
            top_k=args.top_k,
            top_p=args.top_p,
            device=args.device
        )
        
        print(response)
        print(f"{'='*60}\n")
        
        return 0
        
    except Exception as e:
        print(f"\nError during generation: {e}")
        import traceback
        traceback.print_exc()
        return 1


def cmd_interactive(args):
    """Run in interactive mode"""
    checkpoint_path = Path(args.checkpoint)
    tokenizer_path = Path(args.tokenizer)
    
    if not checkpoint_path.exists():
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return 1
    
    if not tokenizer_path.exists():
        print(f"Error: Tokenizer not found: {tokenizer_path}")
        return 1
    
    print(f"Loading model from {checkpoint_path}...")
    
    try:
        print(f"\n{'='*60}")
        print(f"GPTMed Interactive Mode")
        print(f"{'='*60}")
        print(f"Type 'quit' or 'exit' to stop")
        print(f"Type 'help' for available commands")
        print(f"{'='*60}\n")
        
        while True:
            try:
                prompt = input("You: ").strip()
                
                if not prompt:
                    continue
                
                if prompt.lower() in ['quit', 'exit', 'q']:
                    print("Goodbye!")
                    break
                
                if prompt.lower() == 'help':
                    print("\nAvailable commands:")
                    print("  quit/exit - Exit interactive mode")
                    print("  help      - Show this help")
                    print("\nOtherwise, type your medical question and press Enter.\n")
                    continue
                
                # Generate response using gptmed API
                response = gptmed.generate(
                    checkpoint=str(checkpoint_path),
                    tokenizer=str(tokenizer_path),
                    prompt=prompt,
                    max_length=args.max_length,
                    temperature=args.temperature,
                    top_k=args.top_k,
                    top_p=args.top_p,
                    device=args.device
                )
                
                print(f"\nGPTMed: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye!")
                break
            except EOFError:
                print("\nGoodbye!")
                break
        
        return 0
        
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    parser = argparse.ArgumentParser(
        description='GPTMed API Runner - Train and use GPTMed models',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Create a new configuration file
  python run_gptmed.py create-config --output my_config.yaml
  
  # Train a model
  python run_gptmed.py train --config my_config.yaml
  
  # Generate text
  python run_gptmed.py generate \\
      --checkpoint ./model/checkpoints/best_model.pt \\
      --tokenizer medquad_tokenizer.model \\
      --prompt "What is diabetes?"
  
  # Interactive mode
  python run_gptmed.py interactive \\
      --checkpoint ./model/checkpoints/best_model.pt \\
      --tokenizer medquad_tokenizer.model
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    subparsers.required = True
    
    # create-config command
    parser_config = subparsers.add_parser('create-config', help='Create a default configuration file')
    parser_config.add_argument('--output', '-o', default='my_config.yaml',
                             help='Output configuration file (default: my_config.yaml)')
    parser_config.add_argument('--force', '-f', action='store_true',
                             help='Overwrite if file exists')
    parser_config.set_defaults(func=cmd_create_config)
    
    # train command
    parser_train = subparsers.add_parser('train', help='Train a model')
    parser_train.add_argument('--config', '-c', required=True,
                            help='Path to training configuration YAML file')
    parser_train.set_defaults(func=cmd_train)
    
    # generate command
    parser_generate = subparsers.add_parser('generate', help='Generate text from a trained model')
    parser_generate.add_argument('--checkpoint', '-m', required=True,
                               help='Path to model checkpoint (.pt file)')
    parser_generate.add_argument('--tokenizer', '-t', required=True,
                               help='Path to tokenizer model file')
    parser_generate.add_argument('--prompt', '-p', required=True,
                               help='Input prompt for generation')
    parser_generate.add_argument('--max-length', type=int, default=150,
                               help='Maximum length of generated text (default: 150)')
    parser_generate.add_argument('--temperature', type=float, default=0.7,
                               help='Temperature for sampling (default: 0.7)')
    parser_generate.add_argument('--top-k', type=int, default=50,
                               help='Top-k sampling parameter (default: 50)')
    parser_generate.add_argument('--top-p', type=float, default=0.9,
                               help='Top-p (nucleus) sampling parameter (default: 0.9)')
    parser_generate.add_argument('--device', default='cuda',
                               help='Device to use (cuda/cpu, default: cuda)')
    parser_generate.set_defaults(func=cmd_generate)
    
    # interactive command
    parser_interactive = subparsers.add_parser('interactive', help='Run in interactive mode')
    parser_interactive.add_argument('--checkpoint', '-m', required=True,
                                  help='Path to model checkpoint (.pt file)')
    parser_interactive.add_argument('--tokenizer', '-t', required=True,
                                  help='Path to tokenizer model file')
    parser_interactive.add_argument('--max-length', type=int, default=150,
                                  help='Maximum length of generated text (default: 150)')
    parser_interactive.add_argument('--temperature', type=float, default=0.7,
                                  help='Temperature for sampling (default: 0.7)')
    parser_interactive.add_argument('--top-k', type=int, default=50,
                                  help='Top-k sampling parameter (default: 50)')
    parser_interactive.add_argument('--top-p', type=float, default=0.9,
                                  help='Top-p (nucleus) sampling parameter (default: 0.9)')
    parser_interactive.add_argument('--device', default='cuda',
                                  help='Device to use (cuda/cpu, default: cuda)')
    parser_interactive.set_defaults(func=cmd_interactive)
    
    args = parser.parse_args()
    
    # Call the appropriate function
    return args.func(args)


if __name__ == '__main__':
    sys.exit(main())
