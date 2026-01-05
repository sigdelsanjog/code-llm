#!/usr/bin/env python3
"""
Script to pre-download all models locally to models/ folder
"""
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM
import os

# Create models directory if it doesn't exist
os.makedirs("models", exist_ok=True)

models_to_download = [
    {
        "name": "distilgpt2",
        "local_path": "models/distilgpt2",
        "task": "text-generation",
        "model_type": "causal_lm"
    },
    {
        "name": "sshleifer/tiny-gpt2",
        "local_path": "models/tiny-gpt2",
        "task": "text-generation",
        "model_type": "causal_lm"
    },
    {
        "name": "google/t5-efficient-tiny",
        "local_path": "models/t5-efficient-tiny",
        "task": "text2text-generation",
        "model_type": "seq2seq_lm"
    }
]

print("Starting to download models locally...")
for model_config in models_to_download:
    model_name = model_config["name"]
    local_path = model_config["local_path"]
    model_type = model_config["model_type"]
    
    try:
        print(f"\nDownloading {model_name} to {local_path}...")
        
        if model_type == "causal_lm":
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(model_name)
        else:  # seq2seq_lm
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
        # Save locally
        tokenizer.save_pretrained(local_path)
        model.save_pretrained(local_path)
        
        print(f"✓ Successfully downloaded {model_name}")
    except Exception as e:
        print(f"✗ Failed to download {model_name}: {e}")

print("\nAll models downloaded to models/ directory successfully!")
