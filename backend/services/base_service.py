"""
Base service interface for model operations.

Follows SOLID principles:
- Interface Segregation Principle (ISP): Defines minimal interface for model services
- Dependency Inversion Principle (DIP): High-level modules depend on this abstraction
"""

from abc import ABC, abstractmethod
from typing import Dict


class BaseModelService(ABC):
    """
    Abstract base class for model services.
    All model services must implement this interface to ensure consistency
    and allow for polymorphic usage.
    """
    
    @abstractmethod
    def generate(self, prompt: str) -> Dict[str, str]:
        """
        Generate text based on the prompt.
        
        Args:
            prompt: The input prompt text
            
        Returns:
            Dictionary containing:
                - 'model': Name of the model
                - 'response': Generated text or error message
        """
        pass
    
    @abstractmethod
    def get_model_name(self) -> str:
        """
        Get the name identifier of the model.
        
        Returns:
            String identifier for the model
        """
        pass
    
    @abstractmethod
    def is_loaded(self) -> bool:
        """
        Check if the model is loaded and ready to use.
        
        Returns:
            True if model is loaded successfully, False otherwise
        """
        pass
