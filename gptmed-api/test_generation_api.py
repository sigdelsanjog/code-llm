"""
Test GPTMed API - Generation

This script tests the generation function using a trained model.
"""

import gptmed

print("="*60)
print("Testing GPTMed Generation API")
print("="*60)
print(f"\nGPTMed version: {gptmed.__version__}\n")

# Use the trained model (update path if needed)
checkpoint = "./model/checkpoints/best_model.pt"
tokenizer = "../medllm/tokenizer/medquad_tokenizer.model"

prompts = [
    "What is diabetes?",
    "What causes high blood pressure?",
    "How to treat fever?",
    "What are the symptoms of asthma?",
    "How to prevent heart disease?"
]

print(f"Using checkpoint: {checkpoint}")
print(f"Using tokenizer: {tokenizer}\n")

for i, prompt in enumerate(prompts, 1):
    print(f"\n{'='*60}")
    print(f"Test {i}/{len(prompts)}: {prompt}")
    print('='*60)
    
    try:
        answer = gptmed.generate(
            checkpoint=checkpoint,
            tokenizer=tokenizer,
            prompt=prompt,
            max_length=150,
            temperature=0.7,
            top_k=50,
            top_p=0.9,
            device="cuda"
        )
        
        print(f"\n📝 Generated:")
        print(answer)
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

print(f"\n{'='*60}")
print("✅ Generation tests complete!")
print('='*60)
