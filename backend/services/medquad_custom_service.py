"""
MedQuAD Custom Model Service.

Custom GPT model trained from scratch on MedQuAD dataset.
This service integrates the user's custom-trained medical language model
into the backend API following SOLID principles.
"""

from typing import Dict
import torch
import sentencepiece as spm
from pathlib import Path

from .base_service import BaseModelService
from config import get_model_config


class MedQuADCustomService(BaseModelService):
    """
    Service class for custom MedQuAD GPT model.
    Handles loading and inference for the user's trained model.
    """
    
    def __init__(self):
        """Initialize the MedQuAD Custom service and load the model."""
        self._model = None
        self._tokenizer = None
        self._device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._model_name = "MedQuAD-Custom"
        self._config = get_model_config(self._model_name)
        self._load_model()
    
    def _load_model(self):
        """
        Load the custom GPT model and tokenizer.
        Private method following encapsulation principle.
        """
        try:
            model_path = Path(self._config["model_path"]) / "best_model.pt"
            tokenizer_path = Path(self._config["model_path"]) / "medquad_tokenizer.model"
            
            # Load checkpoint
            checkpoint = torch.load(model_path, map_location=self._device)
            
            # Reconstruct model from saved config
            from llm_med.model.architecture import GPTTransformer
            from llm_med.model.configs.model_config import ModelConfig
            
            model_config = ModelConfig(**checkpoint['model_config'])
            self._model = GPTTransformer(model_config)
            self._model.load_state_dict(checkpoint['model_state_dict'])
            self._model.to(self._device)
            self._model.eval()
            
            # Load tokenizer
            self._tokenizer = spm.SentencePieceProcessor()
            self._tokenizer.load(str(tokenizer_path))
            
            print(f"✓ Successfully loaded {self._model_name}")
            print(f"  - Trained steps: {checkpoint.get('step', 'unknown')}")
            print(f"  - Validation loss: {checkpoint.get('val_loss', 'unknown'):.4f}")
            
        except Exception as e:
            print(f"✗ Failed to load {self._model_name}: {e}")
            self._model = None
            self._tokenizer = None
    
    def _generate_text(self, prompt: str, max_length: int = 150, temperature: float = 0.7, 
                       top_k: int = 50, top_p: float = 0.95, repetition_penalty: float = 1.2) -> str:
        """
        Generate text using the custom model.
        
        Args:
            prompt: Input text prompt
            max_length: Maximum tokens to generate
            temperature: Sampling temperature (0.0 = greedy, higher = more random)
            top_k: Top-k sampling parameter
            top_p: Nucleus sampling parameter
            repetition_penalty: Penalty for repeating tokens
            
        Returns:
            Generated text
        """
        # Encode prompt
        prompt_tokens = self._tokenizer.encode_as_ids(prompt)
        generated = prompt_tokens.copy()
        
        with torch.no_grad():
            for _ in range(max_length):
                # Prepare input
                max_seq_len = self._model.config.max_seq_len
                input_ids = generated[-max_seq_len:]
                input_tensor = torch.tensor([input_ids], device=self._device)
                
                # Forward pass
                logits = self._model(input_tensor)
                next_token_logits = logits[0, -1, :]
                
                # Apply repetition penalty
                if repetition_penalty != 1.0:
                    for token_id in set(generated):
                        next_token_logits[token_id] /= repetition_penalty
                
                # Apply temperature
                if temperature > 0:
                    next_token_logits = next_token_logits / temperature
                    
                    # Top-k sampling
                    if top_k > 0:
                        top_k_values, top_k_indices = torch.topk(next_token_logits, min(top_k, next_token_logits.size(-1)))
                        next_token_logits = torch.full_like(next_token_logits, float('-inf'))
                        next_token_logits[top_k_indices] = top_k_values
                    
                    # Top-p (nucleus) sampling
                    if top_p < 1.0:
                        sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                        cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > top_p
                        sorted_indices_to_remove[0] = False  # Keep at least one token
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        next_token_logits[indices_to_remove] = float('-inf')
                    
                    # Sample
                    probs = torch.softmax(next_token_logits, dim=-1)
                    next_token = torch.multinomial(probs, num_samples=1).item()
                else:
                    # Greedy
                    next_token = next_token_logits.argmax().item()
                
                generated.append(next_token)
                
                # Check for EOS or stop tokens
                if next_token == self._tokenizer.eos_id():
                    break
        
        # Decode
        return self._tokenizer.decode_ids(generated)
    
    def generate(self, prompt: str) -> Dict[str, str]:
        """
        Generate medical text using custom MedQuAD model.
        
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
            
            # Use conservative generation settings for medical domain
            generated_text = self._generate_text(
                prompt=prompt,
                max_length=150,
                temperature=0.6,  # Conservative for medical accuracy
                top_k=40,
                top_p=0.9,
                repetition_penalty=1.2
            )
            
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
        return self._model is not None and self._tokenizer is not None
