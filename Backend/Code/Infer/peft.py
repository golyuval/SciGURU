#!/usr/bin/env python3
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# -------------------------------
# 1. Configuration and Paths
# -------------------------------

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Base model name
AUTH_TOKEN = os.getenv("HF_READ_TOKEN")    # Hugging Face auth token

# Path to the DPO adapter weights
ADAPTER_PATH = "Code/Train/RewardModel/DPO/dpo_adapter"

# -------------------------------
# 2. Load the Tokenizer and Model
# -------------------------------

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, token=AUTH_TOKEN)

# Create quantization config with memory optimizations
print("Creating quantization configuration...")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading if needed
    llm_int8_skip_modules=["lm_head"],      # Skip modules from quantization
    llm_int8_threshold=6.0,
    bnb_4bit_compute_dtype=torch.float16
)

print("Loading base model in 8-bit precision...")
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    quantization_config=bnb_config,  # Use the quantization config
    device_map="auto",               # Automatically place model on available devices
    token=AUTH_TOKEN,
    torch_dtype=torch.float16        # Use fp16 for non-quantized weights
)

# Load the fine-tuned adapter weights if they exist
if os.path.exists(ADAPTER_PATH):
    print(f"Loading DPO adapter weights from: {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    print("DPO adapter loaded successfully!")
else:
    print("DPO adapter not found, using base model only.")
    model = base_model

# Set the model to evaluation mode
model.eval()

# Clear CUDA cache to free up memory
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# -------------------------------
# 3. Define the Inference Function
# -------------------------------

def generate_answer(model, prompt, max_new_tokens=300):  # Increased token limit for complete answers
    # Format the prompt according to Llama 3.1 chat template
    formatted_prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant that answers questions accurately and concisely.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n{prompt}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
    
    inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
    input_ids = inputs.input_ids
    attention_mask = inputs.attention_mask
    
    # Generate with memory optimization settings
    with torch.no_grad():  # No need for gradients during inference
        output_ids = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            pad_token_id=tokenizer.eos_token_id,
            use_cache=True  # Enable KV cache for faster generation
        )
    
    # Decode only the newly generated tokens
    new_tokens = output_ids[0, input_ids.shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    # Clean up memory
    del inputs, output_ids, new_tokens
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return answer

# -------------------------------
# 4. Run the Interactive CLI Loop
# -------------------------------

if __name__ == "__main__":
    print("\nModel loaded successfully. You can now ask questions.")
    print("Type 'exit' to quit.\n")
    print(f"Using model: {BASE_MODEL_NAME} " + ("with DPO fine-tuning" if os.path.exists(ADAPTER_PATH) else "without fine-tuning"))

    while True:
        question = input("\nEnter your question: ").strip()
        if question.lower() == "exit":
            print("Exiting interactive session.")
            break
        print("Generating answer...\n")
        answer = generate_answer(model, question)
        print("Answer:", answer)