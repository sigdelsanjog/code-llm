"""
DistilGPT2 Model Service.

Follows SOLID principles:
- Single Responsibility Principle (SRP): Handles only DistilGPT2 model operations
- Open/Closed Principle (OCP): Open for extension through inheritance
- Liskov Substitution Principle (LSP): Can be used anywhere BaseModelService is expected
"""

from typing import Dict
from transformers import pipeline
from .base_service import BaseModelService
from config import get_model_config


class DistilGPT2Service(BaseModelService):
    """
    Service class for DistilGPT2 model operations.
    Encapsulates all DistilGPT2-specific logic and configuration.
    """
    
    def __init__(self):
        """Initialize the DistilGPT2 service and load the model."""
        self._model = None
        self._model_name = "DistilGPT2"
        self._config = get_model_config(self._model_name)
        self._load_model()
    
    def _load_model(self):
        """
        Load the DistilGPT2 model into memory.
        Private method following encapsulation principle.
        """
        try:
            self._model = pipeline(
                self._config["task"],
                model=self._config["model_path"]
            )
            print(f"✓ Successfully loaded {self._model_name}")
        except Exception as e:
            print(f"✗ Failed to load {self._model_name}: {e}")
            self._model = None
    
    def generate(self, prompt: str) -> Dict[str, str]:
        """
        Generate text using DistilGPT2 model.
        
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
            
            # Generate with DistilGPT2-specific parameters
            response = self._model(
                prompt,
                max_length=100,
                num_return_sequences=1,
                do_sample=True,
                temperature=0.7,
                pad_token_id=50256  # Set pad_token_id to eos_token_id
            )
            generated_text = response[0]["generated_text"].strip()
            
            return {
                "model": self._model_name,
                "response": generated_text
            }
        except Exception as e:
            return {
                "model": self._model_name,
                "response": f"Error generating with {self._model_name}: {str(e)}"
            }
    
    def get_model_name(self) -> str:
        """Get the model name identifier."""
        return self._model_name
    
    def is_loaded(self) -> bool:
        """Check if the model is loaded and ready."""
        return self._model is not None
