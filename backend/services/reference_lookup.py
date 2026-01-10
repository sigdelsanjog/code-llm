"""
Reference Answer Lookup Service.

Parses training data Q&A files and provides lookup functionality
to compare model outputs with ground truth answers.
"""

from pathlib import Path
from typing import Optional, Dict
import re


class ReferenceAnswerLookup:
    """
    Lookup service for reference answers from training data.
    Used to compare model outputs with ground truth.
    """
    
    def __init__(self, qa_file_path: str = None):
        """
        Initialize the lookup service.
        
        Args:
            qa_file_path: Path to the Q&A file (input_simple.txt format)
        """
        self._qa_pairs: Dict[str, str] = {}
        self._file_path = qa_file_path
        
        if qa_file_path:
            self._load_qa_file(qa_file_path)
    
    def _normalize_question(self, question: str) -> str:
        """
        Normalize question for matching.
        Removes extra whitespace, converts to lowercase, strips punctuation.
        """
        # Remove Q: prefix if present
        question = re.sub(r'^Q:\s*', '', question, flags=re.IGNORECASE)
        # Lowercase and strip
        question = question.lower().strip()
        # Remove extra whitespace
        question = ' '.join(question.split())
        # Remove trailing question mark and whitespace
        question = question.rstrip('? ')
        return question
    
    def _load_qa_file(self, file_path: str):
        """
        Load and parse Q&A file.
        
        Expected format:
        Q: Question text?
        A: Answer text.
        
        (blank line between pairs)
        """
        try:
            path = Path(file_path)
            if not path.exists():
                print(f"⚠ Reference Q&A file not found: {file_path}")
                return
            
            with open(path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Split by Q: to get individual Q&A pairs
            # Pattern: Q: question\nA: answer
            pattern = r'Q:\s*(.+?)\nA:\s*(.+?)(?=\n\nQ:|\n\nQ\s*:|\Z)'
            matches = re.findall(pattern, content, re.DOTALL)
            
            for question, answer in matches:
                normalized_q = self._normalize_question(question.strip())
                self._qa_pairs[normalized_q] = answer.strip()
            
            print(f"✓ Loaded {len(self._qa_pairs)} Q&A pairs from reference file")
            
        except Exception as e:
            print(f"✗ Error loading Q&A file: {e}")
    
    def find_reference_answer(self, question: str) -> Optional[str]:
        """
        Find the reference answer for a given question.
        
        Args:
            question: The user's question/prompt
            
        Returns:
            The reference answer if found, None otherwise
        """
        normalized_q = self._normalize_question(question)
        
        # Try exact match first
        if normalized_q in self._qa_pairs:
            return self._qa_pairs[normalized_q]
        
        # Try partial match (question contains or is contained by a key)
        for stored_q, answer in self._qa_pairs.items():
            # Check if the normalized question matches closely
            if normalized_q in stored_q or stored_q in normalized_q:
                return answer
            
            # Check word overlap for fuzzy matching
            stored_words = set(stored_q.split())
            query_words = set(normalized_q.split())
            
            # If significant word overlap (>70%), consider it a match
            if stored_words and query_words:
                overlap = len(stored_words & query_words)
                max_len = max(len(stored_words), len(query_words))
                if overlap / max_len > 0.7:
                    return answer
        
        return None
    
    def get_all_questions(self) -> list:
        """Get all questions in the reference dataset."""
        return list(self._qa_pairs.keys())
    
    def get_pair_count(self) -> int:
        """Get the number of Q&A pairs loaded."""
        return len(self._qa_pairs)


# Singleton instance for the GptMed reference data
_gptmed_reference_lookup: Optional[ReferenceAnswerLookup] = None


def get_gptmed_reference_lookup() -> ReferenceAnswerLookup:
    """
    Get the singleton reference lookup instance for GptMed.
    Lazy loads the Q&A file on first access.
    """
    global _gptmed_reference_lookup
    
    if _gptmed_reference_lookup is None:
        # Path to the reference Q&A file
        qa_file = Path(__file__).parent.parent / "models" / "gptmed" / "input_simple.txt"
        _gptmed_reference_lookup = ReferenceAnswerLookup(str(qa_file))
    
    return _gptmed_reference_lookup
