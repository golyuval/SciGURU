#!/usr/bin/env python3
"""
DPO Test Script

This script tests the fine-tuned model with DPO by comparing responses
before and after fine-tuning.
"""

import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import logging
from tqdm import tqdm
import argparse

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Constants
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Base model name
AUTH_TOKEN = os.getenv("HF_READ_TOKEN")  # Hugging Face auth token
ADAPTER_PATH = "Code/Train/RewardModel/DPO/dpo_adapter"
TEST_QUESTIONS_PATH = "Utils/scientific_questions.txt"

def load_questions(file_path):
    """
    Load and parse questions from a text file.
    
    Args:
        file_path: Path to the text file containing questions
        
    Returns:
        List of parsed questions
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Questions file not found at {file_path}")
    
    questions = []
    with open(file_path, 'r') as f:
        for line in f:
            # Parse the line (format: "- Question")
            line = line.strip()
            if line.startswith('-'):
                question = line[1:].strip()  # Remove the leading dash and whitespace
                questions.append(question)
    
    logger.info(f"Loaded {len(questions)} questions from {file_path}")
    return questions

def format_prompt(question):
    """
    Format a question as a prompt for the model.
    
    Args:
        question: The scientific question to format
        
    Returns:
        Formatted prompt string
    """
    # Format according to Llama 3.1 chat template
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant that answers scientific questions accurately and concisely.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def generate_answer(model, tokenizer, prompt, max_new_tokens=150):
    """
    Generate an answer for a given prompt.
    
    Args:
        model: The language model
        tokenizer: The tokenizer
        prompt: The input prompt
        max_new_tokens: Maximum number of tokens to generate
        
    Returns:
        Generated text (answer)
    """
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    # Generate with consistent parameters
    output_ids = model.generate(
        inputs.input_ids,
        attention_mask=inputs.attention_mask,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7,
        top_p=0.9,
        pad_token_id=tokenizer.pad_token_id
    )
    
    # Decode only the newly generated tokens
    new_tokens = output_ids[0, inputs.input_ids.shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    return answer

def main():
    parser = argparse.ArgumentParser(description="Test DPO fine-tuned model")
    parser.add_argument("--num_questions", type=int, default=3, 
                        help="Number of questions to test")
    parser.add_argument("--max_tokens", type=int, default=150,
                        help="Maximum tokens to generate per response")
    args = parser.parse_args()
    
    # 1. Load questions
    try:
        all_questions = load_questions(TEST_QUESTIONS_PATH)
        # Use a subset of questions for testing
        test_questions = all_questions[:args.num_questions]
    except FileNotFoundError:
        logger.error(f"Questions file not found at {TEST_QUESTIONS_PATH}")
        return
    
    # 2. Load the base model without adapters
    logger.info("Loading base model without adapters...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME, 
        use_auth_token=AUTH_TOKEN
    )
    
    # Handle padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        load_in_8bit=True,
        device_map="auto",
        use_auth_token=AUTH_TOKEN
    )
    
    # 3. Generate answers with the base model
    logger.info("Generating answers with the base model...")
    base_model_answers = {}
    
    for question in tqdm(test_questions, desc="Testing base model"):
        prompt = format_prompt(question)
        answer = generate_answer(base_model, tokenizer, prompt, args.max_tokens)
        base_model_answers[question] = answer
    
    # 4. Load the fine-tuned model with adapters
    logger.info(f"Loading fine-tuned model with adapters from {ADAPTER_PATH}...")
    
    if not os.path.exists(ADAPTER_PATH):
        logger.error(f"Adapter not found at {ADAPTER_PATH}. Run the DPO training first.")
        return
    
    # Load the adapter on top of the base model
    fine_tuned_model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    
    # 5. Generate answers with the fine-tuned model
    logger.info("Generating answers with the fine-tuned model...")
    fine_tuned_answers = {}
    
    for question in tqdm(test_questions, desc="Testing fine-tuned model"):
        prompt = format_prompt(question)
        answer = generate_answer(fine_tuned_model, tokenizer, prompt, args.max_tokens)
        fine_tuned_answers[question] = answer
    
    # 6. Display results
    print("\n=== Model Comparison Results ===")
    for question in test_questions:
        print(f"\nQuestion: {question}")
        print("\nBase Model Answer:")
        print(base_model_answers[question])
        print("\nFine-tuned Model Answer:")
        print(fine_tuned_answers[question])
        print("-" * 80)

if __name__ == "__main__":
    main()