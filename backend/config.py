import os
from typing import Dict, List

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Model Configuration
# Centralized configuration for all available models
MODEL_CONFIGS: Dict[str, Dict[str, str]] = {
    "DistilGPT2": {
        "task": "text-generation",
        "model_path": "models/distilgpt2",
        "display_name": "DistilGPT2",
        "description": "Distilled version of GPT2, faster and smaller"
    },
    "GPT2-Tiny": {
        "task": "text-generation", 
        "model_path": "models/tiny-gpt2",
        "display_name": "GPT2-Tiny",
        "description": "Tiny GPT2 model for quick testing"
    },
    "T5-Tiny": {
        "task": "text2text-generation",
        "model_path": "models/t5-efficient-tiny",
        "display_name": "T5-Tiny",
        "description": "Tiny T5 model for text-to-text generation"
    }
}

def get_available_models() -> List[Dict[str, str]]:
    """
    Returns a list of available models with their metadata.
    This follows the design principle of encapsulating model metadata
    and providing a clean interface for model discovery.
    """
    return [
        {
            "id": model_id,
            "display_name": config["display_name"],
            "description": config["description"],
            "task": config["task"]
        }
        for model_id, config in MODEL_CONFIGS.items()
    ]

def get_model_config(model_id: str) -> Dict[str, str]:
    """
    Get configuration for a specific model.
    
    Args:
        model_id: The unique identifier for the model
        
    Returns:
        Model configuration dictionary
        
    Raises:
        ValueError: If model_id is not found
    """
    if model_id not in MODEL_CONFIGS:
        raise ValueError(f"Model '{model_id}' not found. Available models: {list(MODEL_CONFIGS.keys())}")
    return MODEL_CONFIGS[model_id]
