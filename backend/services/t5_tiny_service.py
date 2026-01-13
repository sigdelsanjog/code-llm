"""
T5-Tiny Model Service.

Follows SOLID principles:
- Single Responsibility Principle (SRP): Handles only T5-Tiny model operations
- Open/Closed Principle (OCP): Open for extension through inheritance
- Liskov Substitution Principle (LSP): Can be used anywhere BaseModelService is expected
"""

from typing import Dict
from transformers import pipeline
from .base_service import BaseModelService
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import get_model_config


class T5TinyService(BaseModelService):
    """
    Service class for T5-Tiny model operations.
    Encapsulates all T5-Tiny-specific logic and configuration.
    T5 is a text-to-text model with different input/output handling.
    """
    
    def __init__(self):
        """Initialize the T5-Tiny service and load the model."""
        self._model = None
        self._model_name = "T5-Tiny"
        self._config = get_model_config(self._model_name)
        self._load_model()
    
    def _load_model(self):
        """
        Load the T5-Tiny model into memory.
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
        Generate text using T5-Tiny model.
        T5 requires a task prefix for text-to-text generation.
        
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
            
            # T5 is a text-to-text model, format prompt accordingly
            formatted_prompt = f"Generate pandas code: {prompt}"
            
            # Generate with T5-specific parameters
            response = self._model(
                formatted_prompt,
                max_length=100,
                num_return_sequences=1
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
