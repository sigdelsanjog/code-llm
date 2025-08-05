# utils.py
from transformers import pipeline
import asyncio
import concurrent.futures
from typing import Dict, List

# Define model configurations but don't load them yet
model_configs = {
    "DistilGPT2": {"task": "text-generation", "model": "distilgpt2"},
    "GPT2-Tiny": {"task": "text-generation", "model": "sshleifer/tiny-gpt2"},
    "T5-Tiny": {"task": "text2text-generation", "model": "google/t5-efficient-tiny"}
}

# Global variable to store loaded models
models = {}

def load_models():
    """Load models lazily when needed"""
    global models
    if not models:
        for name, config in model_configs.items():
            try:
                models[name] = pipeline(config["task"], model=config["model"])
                print(f"Successfully loaded {name}")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
                # Create a dummy model that returns an error message
                models[name] = None

def generate_with_model(model_name: str, prompt: str) -> Dict[str, str]:
    """Generate code with a specific model"""
    try:
        # Load models if not already loaded
        if not models:
            load_models()
        
        model = models.get(model_name)
        if model is None:
            return {
                "model": model_name,
                "response": f"Model {model_name} failed to load or is not available"
            }
        
        if model_name == "T5-Tiny":
            # T5 is a text-to-text model, so we need to format the prompt differently
            formatted_prompt = f"Generate pandas code: {prompt}"
            response = model(formatted_prompt, max_length=100, num_return_sequences=1)
            generated_text = response[0]["generated_text"].strip()
        else:
            # For GPT-based models (DistilGPT2 and GPT2-Tiny)
            response = model(prompt, max_length=100, num_return_sequences=1, do_sample=True, temperature=0.7)
            generated_text = response[0]["generated_text"].strip()
        
        return {
            "model": model_name,
            "response": generated_text
        }
    except Exception as e:
        return {
            "model": model_name,
            "response": f"Error generating with {model_name}: {str(e)}"
        }

async def generate_pandas_code(prompt: str) -> Dict[str, List[Dict[str, str]]]:
    """Generate code using all three models in parallel"""
    
    # Use ThreadPoolExecutor to run model inference in parallel
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        # Submit tasks for all models
        future_to_model = {
            executor.submit(generate_with_model, model_name, prompt): model_name 
            for model_name in model_configs.keys()
        }
        
        results = []
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                results.append(result)
            except Exception as e:
                results.append({
                    "model": model_name,
                    "response": f"Error with {model_name}: {str(e)}"
                })
    
    return {"responses": results}
