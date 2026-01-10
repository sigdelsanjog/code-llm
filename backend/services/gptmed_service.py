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
from config import get_model_config


class GptMedService(BaseModelService):
    """
    Service class for custom GptMed model.
    Handles loading and inference for the user's trained model.
    """
    
    def __init__(self):
        """Initialize the GptMed service and load the model."""
        self._generator = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
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
            model_config = ModelConfig(**checkpoint['model_config'])
            model = GPTTransformer(model_config)
            model.load_state_dict(checkpoint['model_state_dict'])
            model.to(self._device)
            model.eval()
            
            # Load tokenizer
            tokenizer = spm.SentencePieceProcessor()
            tokenizer.load(str(tokenizer_path))
            
            # Create generator using gptmed package
            self._generator = TextGenerator(
                model=model,
                tokenizer=tokenizer,
                device=self._device
            )
            
            print(f"✓ Successfully loaded {self._model_name} using gptmed package")
            print(f"  - Trained steps: {checkpoint.get('step', 'unknown')}")
            print(f"  - Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
            print(f"  - Device: {self._device}")
            
        except Exception as e:
            print(f"✗ Failed to load {self._model_name}: {e}")
            import traceback
            traceback.print_exc()
            self._generator = None
    
    def generate(self, prompt: str) -> Dict[str, str]:
        """
        Generate text using custom GptMed model via gptmed package.
        
        Args:
            prompt: Input text prompt
            
        Returns:
            Dictionary with model name and generated response
        """
        try:
            if not self.is_loaded():
                return {
                    "model": self._model_name,
                    "response": f"Model {self._model_name} failed to load or is not available"
                }
            
            # Use the generator from gptmed package
            # Conservative generation settings for quality
            gen_config = GenerationConfig(
                max_length=150,
                temperature=0.6,  # Conservative for quality
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3
            )
            
            generated_text = self._generator.generate(
                prompt=prompt,
                gen_config=gen_config,
                verbose=False
            )
            
            return {
                "model": self._model_name,
                "response": generated_text
            }
            
        except Exception as e:
            import traceback
            traceback.print_exc()
            return {
                "model": self._model_name,
                "response": f"Error generating with {self._model_name}: {str(e)}"
            }
    
    def get_model_name(self) -> str:
        """Get the model name identifier."""
        return self._model_name
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._generator is not None
