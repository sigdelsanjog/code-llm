"""
Data Preprocessing Pipeline

Main script to orchestrate the full preprocessing pipeline:
1. Parse MedQuAD XML files
2. Filter and clean Q&A pairs
3. Format to causal text
4. Save processed data

Usage:
    python preprocess.py --dataset-path ./dataset/MedQuAD --output-dir ./data/processed
"""

import argparse
import json
from pathlib import Path
from typing import Dict, List

from data.parsers import MedQuADParser
from data.parsers.text_formatter import CausalTextFormatter, FormatConfig


def clean_text(text: str) -> str:
    """
    Clean text data.
    
    Args:
        text: Raw text
        
    Returns:
        Cleaned text
        
    Warning: Keep this minimal. Over-aggressive cleaning can:
    - Remove medical terminology
    - Destroy important punctuation
    - Break numbered lists
    """
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Ensure proper spacing after punctuation
    # (but don't break abbreviations like "Dr." or "etc.")
    
    return text.strip()


def validate_qa_pair(question: str, answer: str, min_length: int = 10) -> bool:
    """
    Validate Q&A pair quality.
    
    Args:
        question: Question text
        answer: Answer text
        min_length: Minimum length for both Q and A
        
    Returns:
        True if valid, False otherwise
        
    Validation criteria:
    - Both Q and A must exist
    - Both must meet minimum length
    - Answer shouldn't just repeat question
    """
    if not question or not answer:
        return False
    
    if len(question) < min_length or len(answer) < min_length:
        return False
    
    # Check if answer is just echoing question
    q_words = set(question.lower().split())
    a_words = set(answer.lower().split())
    overlap = len(q_words & a_words) / len(q_words) if q_words else 0
    
    # If >80% overlap, likely just repeating question
    if overlap > 0.8:
        return False
    
    return True


def save_statistics(stats: Dict, output_path: Path):
    """Save dataset statistics to JSON."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print(f"Statistics saved to {output_path}")


def save_processed_text(text: str, output_path: Path):
    """Save processed text corpus."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(text)
    
    # Print statistics
    num_chars = len(text)
    num_words = len(text.split())
    num_lines = text.count('\n')
    
    print(f"\nProcessed corpus saved to {output_path}")
    print(f"  Characters: {num_chars:,}")
    print(f"  Words: {num_words:,}")
    print(f"  Lines: {num_lines:,}")


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess MedQuAD dataset for causal LM training"
    )
    parser.add_argument(
        '--dataset-path',
        type=str,
        default='./dataset/MedQuAD',
        help='Path to MedQuAD dataset root'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./data/processed',
        help='Output directory for processed data'
    )
    parser.add_argument(
        '--format-type',
        type=str,
        choices=['simple', 'minimal', 'structured'],
        default='simple',
        help='Text formatting style'
    )
    parser.add_argument(
        '--skip-validation',
        action='store_true',
        help='Skip Q&A pair validation'
    )
    parser.add_argument(
        '--min-length',
        type=int,
        default=10,
        help='Minimum length for Q&A pairs'
    )
    
    args = parser.parse_args()
    
    # Setup paths
    dataset_path = Path(args.dataset_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("MedQuAD Preprocessing Pipeline")
    print("=" * 60)
    print(f"Dataset: {dataset_path}")
    print(f"Output: {output_dir}")
    print(f"Format: {args.format_type}")
    print()
    
    # Step 1: Parse XML files
    print("Step 1: Parsing XML files...")
    parser_obj = MedQuADParser(dataset_path, skip_empty=True)
    qa_pairs = parser_obj.parse_all(filter_empty_answers=True)
    
    # Step 2: Get statistics
    print("\nStep 2: Analyzing dataset...")
    stats = parser_obj.get_statistics(qa_pairs)
    print(f"Total pairs: {stats['total_pairs']}")
    print(f"Pairs with answers: {stats['pairs_with_answers']}")
    
    print("\nBreakdown by source:")
    for source, count in sorted(stats['sources'].items()):
        print(f"  {source}: {count}")
    
    print("\nBreakdown by question type:")
    top_qtypes = sorted(stats['qtypes'].items(), key=lambda x: x[1], reverse=True)[:10]
    for qtype, count in top_qtypes:
        print(f"  {qtype}: {count}")
    
    save_statistics(stats, output_dir / 'dataset_stats.json')
    
    # Step 3: Filter and validate
    if not args.skip_validation:
        print("\nStep 3: Validating Q&A pairs...")
        valid_pairs = []
        for pair in qa_pairs:
            question = clean_text(pair.question)
            answer = clean_text(pair.answer)
            
            if validate_qa_pair(question, answer, min_length=args.min_length):
                pair.question = question
                pair.answer = answer
                valid_pairs.append(pair)
        
        print(f"Valid pairs: {len(valid_pairs)} / {len(qa_pairs)}")
        print(f"Filtered out: {len(qa_pairs) - len(valid_pairs)}")
        qa_pairs = valid_pairs
    
    # Step 4: Format to causal text
    print("\nStep 4: Formatting to causal text...")
    
    if args.format_type == 'minimal':
        from data.parsers.text_formatter import MinimalFormatter
        formatter = MinimalFormatter()
    elif args.format_type == 'structured':
        from data.parsers.text_formatter import StructuredFormatter
        formatter = StructuredFormatter()
    else:  # simple
        formatter = CausalTextFormatter(FormatConfig())
    
    formatted_text = formatter.format_from_structured(qa_pairs)
    
    # Step 5: Save processed data
    print("\nStep 5: Saving processed data...")
    output_file = output_dir / f'medquad_{args.format_type}.txt'
    save_processed_text(formatted_text, output_file)
    
    # Also save as JSONL for later analysis
    jsonl_file = output_dir / 'medquad_pairs.jsonl'
    with open(jsonl_file, 'w', encoding='utf-8') as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair.to_dict(), ensure_ascii=False) + '\n')
    print(f"JSONL data saved to {jsonl_file}")
    
    print("\n" + "=" * 60)
    print("✅ Preprocessing complete!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Train SentencePiece tokenizer on processed text")
    print("2. Tokenize dataset")
    print("3. Create training/validation splits")


if __name__ == '__main__':
    main()
