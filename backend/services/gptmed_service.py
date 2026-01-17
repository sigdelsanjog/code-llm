"""
GptMed Custom Model Service.

Custom GPT model trained from scratch using gptmed package.
This service integrates the user's custom-trained language model
into the backend API following SOLID principles.

Now uses the gptmed package from PyPI.
"""

from typing import Dict
import torch
from pathlib import Path

# Import from llm-med package v0.2.0
from gptmed.model.architecture import GPTTransformer
from gptmed.model.configs.model_config import ModelConfig
from gptmed.inference.generator import TextGenerator
from gptmed.inference.generation_config import GenerationConfig
import sentencepiece as spm

from .base_service import BaseModelService
from .reference_lookup import get_gptmed_reference_lookup
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_model_config


class GptMedService(BaseModelService):
    """
    Service class for custom GptMed model.
    Handles loading and inference for the user's trained model.
    """
    
    def __init__(self):
        """Initialize the GptMed service and load the model."""
        self._generator = None
        # Force CPU since GTX 1060 (sm_61) is incompatible with PyTorch 2.9.1
        # PyTorch 2.9.1 requires CUDA capability >= sm_70
        self._device = 'cpu'  # Can be overridden to 'cuda' if compatible GPU available
        self._model_name = "GptMed"
        self._config = get_model_config(self._model_name)
        self._load_model()
    
    def _load_model(self):
        """
        Load the custom GPT model and tokenizer using gptmed package.
        Private method following encapsulation principle.
        """
        try:
            model_path = Path(self._config["model_path"]) / "gptmed_model.pt"
            tokenizer_path = Path(self._config["model_path"]) / "gptmed_tokenizer.model"
            
            # Check if files exist
            if not model_path.exists():
                raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
            if not tokenizer_path.exists():
                raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self._device)
            
            # Reconstruct model from saved config using gptmed package
            model_config_data = checkpoint.get("model_config")
            if model_config_data is None:
                raise KeyError("Checkpoint missing required key: 'model_config'")

            # gptmed checkpoints typically store a dict from ModelConfig.to_dict()
            model_config = (
                ModelConfig.from_dict(model_config_data)
                if isinstance(model_config_data, dict)
                else ModelConfig(**model_config_data)
            )
            model = GPTTransformer(model_config)
            state_dict = checkpoint.get("model_state_dict")
            if state_dict is None:
                raise KeyError("Checkpoint missing required key: 'model_state_dict'")

            model.load_state_dict(state_dict)
            model.to(self._device)
            model.eval()
            
            # Load tokenizer
            tokenizer = spm.SentencePieceProcessor()
            # SentencePiece uses Load() (capital L)
            tokenizer.Load(str(tokenizer_path))
            
            # Create generator using gptmed package
            self._generator = TextGenerator(
                model=model,
                tokenizer=tokenizer,
                device=self._device
            )
            
            print(f"✓ Successfully loaded {self._model_name} using gptmed package")
            trained_steps = checkpoint.get("step")
            val_loss = checkpoint.get("val_loss")
            print(f"  - Trained steps: {trained_steps if trained_steps is not None else 'unknown'}")
            if isinstance(val_loss, (int, float)):
                print(f"  - Validation loss: {val_loss:.4f}")
            else:
                print("  - Validation loss: unknown")
            print(f"  - Device: {self._device}")
            
        except Exception as e:
            print(f"✗ Failed to load {self._model_name}: {e}")
            import traceback
            traceback.print_exc()
            self._generator = None
    
    def _normalize_prompt(self, prompt: str) -> str:
        """
        Normalize user prompt to match training data format.
        
        Training data uses format: "Q: What is (are) X ?" or "Q: What are the symptoms of X ?"
        """
        import re
        
        # Remove leading/trailing whitespace
        prompt = prompt.strip()
        
        # If already starts with Q:, keep it
        if prompt.upper().startswith("Q:"):
            return prompt
        
        # Common corrections for question format
        # "Who are" -> "What are" (common typo)
        prompt = re.sub(r'^[Ww]ho\s+are', 'What are', prompt)
        
        # Add "Q: " prefix if not present
        if not prompt.startswith("Q:"):
            prompt = f"Q: {prompt}"
        
        # Ensure question mark with space before it (training data style)
        if not prompt.endswith("?"):
            prompt = prompt.rstrip(".") + " ?"
        elif prompt.endswith("?") and not prompt.endswith(" ?"):
            prompt = prompt[:-1] + " ?"
        
        return prompt

    def generate(self, prompt: str) -> Dict[str, str]:
        """
        Generate text using custom GptMed model via gptmed package.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Dictionary with model name, generated response, and reference answer
        """
        try:
            # Normalize prompt to match training data format
            normalized_prompt = self._normalize_prompt(prompt)
            
            # Look up reference answer from training data
            reference_lookup = get_gptmed_reference_lookup()
            reference_answer = reference_lookup.find_reference_answer(prompt)
            
            if not self.is_loaded():
                return {
                    "model": self._model_name,
                    "response": f"Model {self._model_name} failed to load or is not available",
                    "reference_answer": reference_answer
                }
            
            # Use the generator from gptmed package
            # Conservative generation settings for quality
            gen_config = GenerationConfig(
                max_length=200,  # Increased for longer answers
                temperature=0.5,  # Lower for more deterministic output
                top_k=30,         # More focused sampling
                top_p=0.85,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3
            )
            
            generated_text = self._generator.generate(
                prompt=normalized_prompt,
                gen_config=gen_config,
                verbose=False
            )
            
            # Clean up output - remove the prompt from response if echoed
            if generated_text.startswith(normalized_prompt):
                generated_text = generated_text[len(normalized_prompt):].strip()
            
            # Remove "A: " prefix if present
            if generated_text.startswith("A:"):
                generated_text = generated_text[2:].strip()
            
            return {
                "model": self._model_name,
                "response": generated_text,
                "reference_answer": reference_answer
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            
            # Still try to get reference answer even on error
            reference_answer = None
            try:
                reference_lookup = get_gptmed_reference_lookup()
                reference_answer = reference_lookup.find_reference_answer(prompt)
            except:
                pass
            
            return {
                "model": self._model_name,
                "response": f"Error generating with {self._model_name}: {str(e)}",
                "reference_answer": reference_answer
            }
    
    def get_model_name(self) -> str:
        """Get the model name identifier."""
        return self._model_name
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._generator is not None
