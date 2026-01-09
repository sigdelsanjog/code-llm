"""
Service Factory for creating model service instances.

Follows SOLID principles:
- Single Responsibility Principle (SRP): Responsible only for creating service instances
- Open/Closed Principle (OCP): Can add new services without modifying existing code
- Dependency Inversion Principle (DIP): Returns abstract BaseModelService type
"""

from typing import Dict, List, Optional
from .base_service import BaseModelService
from .distilgpt2_service import DistilGPT2Service
from .gpt2_tiny_service import GPT2TinyService
from .t5_tiny_service import T5TinyService
from .gptgpt_service import GptGptService


class ServiceFactory:
    """
    Factory class for creating and managing model service instances.
    Implements lazy loading and singleton pattern for services.
    """
    
    # Class-level cache for service instances (singleton per service type)
    _service_instances: Dict[str, BaseModelService] = {}
    
    # Mapping of model names to service classes
    _service_registry: Dict[str, type] = {
        "DistilGPT2": DistilGPT2Service,
        "GPT2-Tiny": GPT2TinyService,
        "T5-Tiny": T5TinyService,
        "GptGpt": GptGptService
    }
    
    @classmethod
    def get_service(cls, model_name: str) -> BaseModelService:
        """
        Get or create a service instance for the specified model.
        Uses lazy loading and caching for efficiency.
        
        Args:
            model_name: Name of the model (e.g., "DistilGPT2", "GPT2-Tiny", "T5-Tiny")
            
        Returns:
            Instance of the requested model service
            
        Raises:
            ValueError: If model_name is not registered
        """
        if model_name not in cls._service_registry:
            raise ValueError(
                f"Model '{model_name}' not found. "
                f"Available models: {list(cls._service_registry.keys())}"
            )
        
        # Return cached instance if exists, otherwise create new one
        if model_name not in cls._service_instances:
            service_class = cls._service_registry[model_name]
            cls._service_instances[model_name] = service_class()
        
        return cls._service_instances[model_name]
    
    @classmethod
    def get_all_services(cls) -> List[BaseModelService]:
        """
        Get instances of all registered model services.
        Useful when user selects "All Models" option.
        
        Returns:
            List of all model service instances
        """
        return [
            cls.get_service(model_name) 
            for model_name in cls._service_registry.keys()
        ]
    
    @classmethod
    def get_available_model_names(cls) -> List[str]:
        """
        Get list of all available model names.
        
        Returns:
            List of registered model names
        """
        return list(cls._service_registry.keys())
    
    @classmethod
    def register_service(cls, model_name: str, service_class: type):
        """
        Register a new service class.
        Allows extending the factory with new services (Open/Closed Principle).
        
        Args:
            model_name: Unique identifier for the model
            service_class: Class that implements BaseModelService
            
        Raises:
            TypeError: If service_class doesn't inherit from BaseModelService
        """
        if not issubclass(service_class, BaseModelService):
            raise TypeError(
                f"{service_class.__name__} must inherit from BaseModelService"
            )
        
        cls._service_registry[model_name] = service_class
        
    @classmethod
    def clear_cache(cls):
        """
        Clear all cached service instances.
        Useful for testing or when models need to be reloaded.
        """
        cls._service_instances.clear()
