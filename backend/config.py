"""
Configuration module using Singleton pattern.

Ensures only one instance of configuration exists throughout the application.
Follows SOLID principles and Singleton design pattern.
"""

import os
from typing import Dict, List, Optional


class Config:
    """
    Singleton configuration class for application settings.
    
    Ensures only one configuration instance exists and provides
    centralized access to all configuration data.
    """
    
    _instance: Optional['Config'] = None
    _initialized: bool = False
    
    def __new__(cls):
        """
        Create or return the single instance of Config.
        Implements the Singleton pattern.
        """
        if cls._instance is None:
            cls._instance = super(Config, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """
        Initialize configuration only once.
        Subsequent calls to __init__ will not reinitialize.
        """
        if Config._initialized:
            return
        
        # API Keys
        self.OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
        
        # Model Configuration
        # Centralized configuration for all available models
        self.MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
            "DistilGPT2": {
                "task": "text-generation",
                "model_path": "backend/models/distilgpt2",
                "display_name": "DistilGPT2",
                "description": "Distilled version of GPT2, faster and smaller"
            },
            "GPT2-Tiny": {
                "task": "text-generation", 
                "model_path": "backend/models/tiny-gpt2",
                "display_name": "GPT2-Tiny",
                "description": "Tiny GPT2 model for quick testing"
            },
            "T5-Tiny": {
                "task": "text2text-generation",
                "model_path": "backend/models/t5-efficient-tiny",
                "display_name": "T5-Tiny",
                "description": "Tiny T5 model for text-to-text generation"
            },
            "GptMed": {
                "task": "text-generation",
                "model_path": "backend/models/gptmed",
                "display_name": "GptMed",
                "description": "Custom GPT model trained using gptmed package"
            }
        }
        
        Config._initialized = True
    
    def get_available_models(self) -> List[Dict[str, str]]:
        """
        Returns a list of available models with their metadata.
        This follows the design principle of encapsulating model metadata
        and providing a clean interface for model discovery.
        
        Returns:
            List of dictionaries containing model metadata
        """
        return [
            {
                "id": model_id,
                "display_name": config["display_name"],
                "description": config["description"],
                "task": config["task"]
            }
            for model_id, config in self.MODEL_CONFIGS.items()
        ]
    
    def get_model_config(self, model_id: str) -> Dict[str, str]:
        """
        Get configuration for a specific model.
        
        Args:
            model_id: The unique identifier for the model
            
        Returns:
            Model configuration dictionary
            
        Raises:
            ValueError: If model_id is not found
        """
        if model_id not in self.MODEL_CONFIGS:
            raise ValueError(
                f"Model '{model_id}' not found. "
                f"Available models: {list(self.MODEL_CONFIGS.keys())}"
            )
        return self.MODEL_CONFIGS[model_id]
    
    def get_model_ids(self) -> List[str]:
        """
        Get list of all available model IDs.
        
        Returns:
            List of model identifier strings
        """
        return list(self.MODEL_CONFIGS.keys())
    
    @classmethod
    def reset_instance(cls):
        """
        Reset the singleton instance.
        Useful for testing purposes.
        """
        cls._instance = None
        cls._initialized = False


# Create singleton instance
_config = Config()

# Backward compatibility: expose functions at module level
def get_available_models() -> List[Dict[str, str]]:
    """Get available models from singleton config instance."""
    return _config.get_available_models()


def get_model_config(model_id: str) -> Dict[str, str]:
    """Get model configuration from singleton config instance."""
    return _config.get_model_config(model_id)


# Export singleton instance for direct access
config = _config
