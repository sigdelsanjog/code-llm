# utils.py
from transformers import pipeline

# Initialize the code generation model
generator = pipeline("text-generation", model="EleutherAI/gpt-neo-125M")

def generate_pandas_code(prompt: str) -> str:
    # Generate code based on the prompt
    response = generator(prompt, max_length=100)
    
    # Extract and return the generated code
    code = response[0]["generated_text"].strip()
    return code
