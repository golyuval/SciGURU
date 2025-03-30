#!/usr/bin/env python3
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------------
# 1. Configuration and Paths
# -------------------------------

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Base model name
AUTH_TOKEN = "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"    # Hugging Face auth token

# -------------------------------
# 2. Load the Tokenizer and Model
# -------------------------------

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_auth_token=AUTH_TOKEN)

print("Loading base model in 8-bit precision...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    load_in_8bit=True,         # Enable 8-bit quantization for memory savings
    device_map="auto",         # Automatically place model on available devices
    use_auth_token=AUTH_TOKEN
)

# -------------------------------
# 3. Define the Inference Function
# -------------------------------

def generate_answer(model, prompt, max_new_tokens=150):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    output_ids = model.generate(
        input_ids,
        attention_mask=attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        pad_token_id=tokenizer.eos_token_id  # explicitly set pad_token_id
    )
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text

# -------------------------------
# 4. Run the Interactive CLI Loop
# -------------------------------

if __name__ == "__main__":
    print("\nModel loaded successfully. You can now ask questions.")
    print("Type 'exit' to quit.\n")
    while True:
        question = input("Enter your question: ").strip()
        if question.lower() == "exit":
            print("Exiting interactive session.")
            break
        print("Generating answer...\n")
        answer = generate_answer(base_model, question)
        print("Answer:", answer)
