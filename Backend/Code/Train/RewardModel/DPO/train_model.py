#!/usr/bin/env python3
"""
Memory-Efficient DPO (Direct Preference Optimization) Implementation for Llama 3.1

This script implements DPO fine-tuning for a Llama 3.1 model using human preferences,
optimized for systems with limited GPU memory.
"""

import os
import torch
import numpy as np
import json
import gc
from datetime import datetime
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    BitsAndBytesConfig
)
from peft import (
    LoraConfig, 
    PeftModel, 
    get_peft_model, 
    prepare_model_for_kbit_training
)
from datasets import Dataset
from tqdm import tqdm
import logging
import psutil

# Configure basic logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# -------------------------------
# 1. Configuration and Paths
# -------------------------------

BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Base model name
AUTH_TOKEN = os.getenv("HF_READ_TOKEN")  # Hugging Face auth token

# Paths
QUESTIONS_PATH = "Utils/scientific_questions.txt"
ADAPTER_PATH = "Code/Train/RewardModel/DPO/dpo_adapter"
DATASET_PATH = "Code/Train/RewardModel/DPO/preference_data.json"
OFFLOAD_FOLDER = "offload_folder"  # For CPU offloading

# DPO Hyperparameters - Memory-optimized settings
BETA = 0.1  # Controls the strength of the KL penalty
LEARNING_RATE = 2e-5  # Reduced learning rate for stability (ORIGINAL = 1e-5)
NUM_EPOCHS = 2  # Reduced epochs to save time and memory
BATCH_SIZE = 1  # Single example per batch for minimal memory usage
GRADIENT_ACCUMULATION_STEPS = 4  # Simulate larger batch sizes
MAX_SEQ_LENGTH = 256  # Limit sequence length to save memory

# Create offload directory
os.makedirs(OFFLOAD_FOLDER, exist_ok=True)

# LoRA Configuration - Memory-efficient settings
lora_config = LoraConfig(
    r=8,  # Reduced rank (8 instead of 16)
    lora_alpha=16,  # Scaled alpha
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj",  # Only target critical attention modules
        "v_proj",
        "o_proj",
    ],
)

# -------------------------------
# 2. Utility Functions
# -------------------------------

def log_memory_usage(stage=""):
    """Log current memory usage for debugging."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"{stage} - GPU Memory: {allocated:.2f}MB allocated, {reserved:.2f}MB reserved")
    
    process = psutil.Process()
    ram = process.memory_info().rss / 1024**2
    logger.info(f"{stage} - RAM Usage: {ram:.2f}MB")

def load_questions(file_path):
    """Load and parse questions from a text file."""
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
    """Format a question as a prompt for the model."""
    # Format according to Llama 3.1 chat template
    prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\nYou are a helpful assistant that answers scientific questions accurately and concisely.<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n"
    return prompt

def save_preference_data(data, file_path):
    """Save preference data to a JSON file."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info(f"Saved preference data to {file_path}")

def load_preference_data(file_path):
    """Load preference data from a JSON file."""
    if not os.path.exists(file_path):
        return []
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    logger.info(f"Loaded {len(data)} preference examples from {file_path}")
    return data

def prepare_datasets(preference_data):
    """Prepare datasets for DPO training."""
    formatted_data = {
        "prompt": [],
        "chosen": [],
        "rejected": [],
    }
    
    for item in preference_data:
        formatted_data["prompt"].append(item["prompt"])
        formatted_data["chosen"].append(item["chosen"])
        formatted_data["rejected"].append(item["rejected"])
    
    # Create a Hugging Face dataset
    dataset = Dataset.from_dict(formatted_data)
    return dataset

# -------------------------------
# 3. Model Loading
# -------------------------------

def load_model_and_tokenizer():
    """Load the base model and tokenizer with memory optimizations."""
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL_NAME, 
        token=AUTH_TOKEN
    )
    
    # Handle padding token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Create quantization config with CPU offloading enabled
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_enable_fp32_cpu_offload=True,  # Enable CPU offloading
        llm_int8_skip_modules=["lm_head"],      # Skip modules from quantization
        llm_int8_threshold=6.0,
        bnb_4bit_compute_dtype=torch.float16
    )
    
    logger.info("Loading base model with optimized 8-bit quantization...")
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
        token=AUTH_TOKEN,
        torch_dtype=torch.float16,
        offload_folder=OFFLOAD_FOLDER
    )
    
    # Prepare model for kbit training
    model = prepare_model_for_kbit_training(model)
    
    # Check if adapter exists and load it
    if os.path.exists(ADAPTER_PATH):
        logger.info(f"Loading existing adapter from {ADAPTER_PATH}")
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)
    else:
        # Add LoRA adapter
        logger.info("Adding new LoRA adapter to the model")
        model = get_peft_model(model, lora_config)
    
    # Memory optimization: Clear CUDA cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return tokenizer, model

# -------------------------------
# 4. Generation Functions
# -------------------------------

def generate_answers(model, tokenizer, prompt, num_answers=2, max_new_tokens=300):
    """Generate multiple answers for a single prompt with memory optimizations."""
    generated_answers = []
    
    # Create inputs once to save memory
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    for i in range(num_answers):
        # Different temperature for each generation to ensure diversity
        temp = 0.7 if i == 0 else 0.9
        
        # Generate with memory optimization
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=temp,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
            )
        
        # Decode only the newly generated tokens
        new_tokens = output_ids[0, inputs.input_ids.shape[1]:]
        answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
        generated_answers.append(answer)
        
        # Clear cache between generations
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    # Clean up memory
    del inputs
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    return generated_answers

# -------------------------------
# 5. DPO Training Implementation
# -------------------------------

def dpo_loss(policy_chosen_logps, policy_rejected_logps, 
             reference_chosen_logps, reference_rejected_logps, beta):
    """Calculate the DPO loss."""
    # Calculate the log likelihood ratio for chosen and rejected completions
    chosen_rewards = policy_chosen_logps - reference_chosen_logps
    rejected_rewards = policy_rejected_logps - reference_rejected_logps
    
    # Calculate the DPO loss (Equation 3 in the DPO paper)
    logits = beta * (chosen_rewards - rejected_rewards)
    losses = -torch.nn.functional.logsigmoid(logits)
    
    # Calculate accuracy
    accuracies = (chosen_rewards > rejected_rewards).float()
    
    # Average rewards for logging
    chosen_rewards_mean = chosen_rewards.mean()
    rejected_rewards_mean = rejected_rewards.mean()
    
    return losses.mean(), accuracies.mean(), {
        "chosen_rewards": chosen_rewards_mean.item(),
        "rejected_rewards": rejected_rewards_mean.item(),
        "reward_gap": (chosen_rewards_mean - rejected_rewards_mean).item()
    }

class DPOTrainer:
    """Trainer class for DPO fine-tuning with memory optimizations."""
    def __init__(
        self, 
        model, 
        tokenizer, 
        beta=0.1,
        learning_rate=1e-5,
        num_epochs=2,
        batch_size=1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.beta = beta
        self.learning_rate = learning_rate
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.steps = 0
        
        # We use the same model for policy and reference to save memory
        # The reference logits will be computed with torch.no_grad()
        self.policy_mode = True
        
        # Only optimize trainable parameters (LoRA)
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Use AdamW with weight decay and gradient clipping
        from torch.optim import AdamW
        self.optimizer = AdamW(
            trainable_params,
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.01
        )
        
    def compute_logps(self, input_ids, attention_mask, labels, reference_mode=False):
        """Compute log probabilities with memory optimizations."""
        # For reference model computations, use no_grad
        if reference_mode:
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True
                )
        else:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
        
        logits = outputs.logits
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Calculate log probabilities
        log_probs = torch.nn.functional.log_softmax(shift_logits, dim=-1)
        
        # Gather log probs at label positions
        selected_log_probs = log_probs.gather(
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask and normalize
        shift_mask = attention_mask[..., 1:].contiguous()
        selected_log_probs = selected_log_probs * shift_mask
        
        sequence_lengths = shift_mask.sum(dim=1)
        sequence_log_probs = selected_log_probs.sum(dim=1) / sequence_lengths
        
        return sequence_log_probs
    
    def process_batch(self, batch, reference_mode=False):
        """Process a batch with memory optimizations."""
        # Process prompts
        prompt_tokens = self.tokenizer(
            batch["prompt"], 
            padding=True, 
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Process chosen completions
        chosen_tokens = self.tokenizer(
            batch["chosen"], 
            padding=True, 
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Process rejected completions
        rejected_tokens = self.tokenizer(
            batch["rejected"], 
            padding=True, 
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            return_tensors="pt"
        ).to(self.model.device)
        
        # Compute log probs for chosen completions
        chosen_logps = self.compute_logps(
            chosen_tokens.input_ids,
            chosen_tokens.attention_mask,
            chosen_tokens.input_ids,
            reference_mode
        )
        
        # Free up memory
        del prompt_tokens
        
        # Compute log probs for rejected completions
        rejected_logps = self.compute_logps(
            rejected_tokens.input_ids,
            rejected_tokens.attention_mask,
            rejected_tokens.input_ids,
            reference_mode
        )
        
        # Free up memory
        del chosen_tokens, rejected_tokens
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        return chosen_logps, rejected_logps
    
    def train_step(self, batch):
        """Perform a single training step with memory optimizations."""
        # First get reference model logits (with no_grad)
        ref_chosen_logps, ref_rejected_logps = self.process_batch(batch, reference_mode=True)
        
        # Then get policy model logits (with grad)
        self.model.train()
        policy_chosen_logps, policy_rejected_logps = self.process_batch(batch, reference_mode=False)
        
        # Compute DPO loss
        loss, accuracy, rewards = dpo_loss(
            policy_chosen_logps, 
            policy_rejected_logps,
            ref_chosen_logps,
            ref_rejected_logps,
            self.beta
        )
        
        # Scale the loss for gradient accumulation
        loss = loss / GRADIENT_ACCUMULATION_STEPS
        
        # Update weights
        loss.backward()
        
        # Only update every GRADIENT_ACCUMULATION_STEPS
        if ((self.steps + 1) % GRADIENT_ACCUMULATION_STEPS == 0):
            # Apply gradient clipping
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad], 
                max_norm=1.0
            )
            self.optimizer.step()
            self.optimizer.zero_grad()
        
        # Free up memory
        del ref_chosen_logps, ref_rejected_logps
        del policy_chosen_logps, policy_rejected_logps
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        # Track steps
        self.steps += 1
        
        return loss.item() * GRADIENT_ACCUMULATION_STEPS, accuracy.item(), rewards
    
    def train(self, dataset):
        """Train the model on a dataset of preferences."""
        self.model.train()
        
        all_metrics = {
            "loss": [],
            "accuracy": [],
            "chosen_rewards": [],
            "rejected_rewards": [],
            "reward_gap": [],
        }
        
        # Start with cleared gradients
        self.optimizer.zero_grad()
        
        for epoch in range(self.num_epochs):
            logger.info(f"Starting epoch {epoch+1}/{self.num_epochs}")
            
            # Shuffle dataset each epoch
            dataset = dataset.shuffle()
            
            # Initialize accumulated metrics
            accumulated_loss = 0
            accumulated_accuracy = 0
            accumulated_rewards = {"chosen_rewards": 0, "rejected_rewards": 0, "reward_gap": 0}
            accumulation_count = 0
            
            progress_bar = tqdm(range(0, len(dataset), self.batch_size), desc=f"Epoch {epoch+1}")
            
            for i in progress_bar:
                # Get batch (single example for memory efficiency)
                batch = dataset[i:i+self.batch_size]
                
                # Perform training step
                loss, accuracy, rewards = self.train_step(batch)
                
                # Accumulate metrics
                accumulated_loss += loss
                accumulated_accuracy += accuracy
                accumulated_rewards["chosen_rewards"] += rewards["chosen_rewards"]
                accumulated_rewards["rejected_rewards"] += rewards["rejected_rewards"]
                accumulated_rewards["reward_gap"] += rewards["reward_gap"]
                accumulation_count += 1
                
                # Log metrics and reset accumulators after gradient update
                if ((i // self.batch_size) + 1) % GRADIENT_ACCUMULATION_STEPS == 0 or i + self.batch_size >= len(dataset):
                    # Normalize accumulated values
                    normalized_loss = accumulated_loss / accumulation_count
                    normalized_accuracy = accumulated_accuracy / accumulation_count
                    normalized_rewards = {
                        k: v / accumulation_count for k, v in accumulated_rewards.items()
                    }
                    
                    # Record metrics
                    all_metrics["loss"].append(normalized_loss)
                    all_metrics["accuracy"].append(normalized_accuracy)
                    all_metrics["chosen_rewards"].append(normalized_rewards["chosen_rewards"])
                    all_metrics["rejected_rewards"].append(normalized_rewards["rejected_rewards"])
                    all_metrics["reward_gap"].append(normalized_rewards["reward_gap"])
                    
                    # Update progress bar
                    progress_bar.set_postfix({
                        "loss": f"{normalized_loss:.4f}", 
                        "acc": f"{normalized_accuracy:.4f}",
                        "gap": f"{normalized_rewards['reward_gap']:.4f}"
                    })
                    
                    # Reset accumulators
                    accumulated_loss = 0
                    accumulated_accuracy = 0
                    accumulated_rewards = {"chosen_rewards": 0, "rejected_rewards": 0, "reward_gap": 0}
                    accumulation_count = 0
                    
                    # Clear memory
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    gc.collect()
            
            # Save checkpoint after each epoch
            self.save_model(f"{ADAPTER_PATH}_epoch{epoch+1}")
            logger.info(f"Saved checkpoint for epoch {epoch+1}")
            
            # Clear memory between epochs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
        
        # Calculate average metrics
        avg_metrics = {k: sum(v) / len(v) for k, v in all_metrics.items() if v}
        
        logger.info(f"Training completed. Average metrics: {avg_metrics}")
        return avg_metrics
    
    def save_model(self, path):
        """Save the fine-tuned model adapter."""
        self.model.save_pretrained(path)
        logger.info(f"Model adapter saved to {path}")

# -------------------------------
# 6. Evaluation
# -------------------------------

def evaluate_model(model, tokenizer, questions, max_new_tokens=150):
    """Evaluate the model with memory optimizations."""
    model.eval()
    results = {}
    
    for question in tqdm(questions, desc="Evaluating model"):
        prompt = format_prompt(question)
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            output_ids = model.generate(
                inputs.input_ids,
                attention_mask=inputs.attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True
            )
            
            # Decode only the newly generated tokens
            new_tokens = output_ids[0, inputs.input_ids.shape[1]:]
            answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
            results[question] = answer.strip()
        
        # Clean up for next question
        del inputs, output_ids, new_tokens
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    
    return results

# -------------------------------
# 7. Main Function
# -------------------------------

def main():
    """Main function to run the DPO pipeline."""
    # Log initial memory usage
    log_memory_usage("Initial")
    
    # 1. Load questions
    try:
        questions = load_questions(QUESTIONS_PATH)
    except FileNotFoundError:
        logger.error(f"Questions file not found at {QUESTIONS_PATH}")
        return
    
    # 2. Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()
    
    # Log memory after model load
    log_memory_usage("After model load")
    
    # 3. Check for existing preference data
    preference_data = load_preference_data(DATASET_PATH)
    
    # 4. Collect preferences if needed
    if len(preference_data) < len(questions):
        logger.info("Collecting preferences for questions...")
        
        for question in questions:
            # Skip questions we already have preferences for
            if any(item["prompt"] == format_prompt(question) for item in preference_data):
                continue
                
            print(f"\nQuestion: {question}")
            
            # Generate two different answers
            prompt = format_prompt(question)
            answers = generate_answers(model, tokenizer, prompt, num_answers=2)
            
            print("\nAnswer 1:")
            print(answers[0])
            print("\nAnswer 2:")
            print(answers[1])
            
            # Ask for preference
            while True:
                choice = input("\nWhich answer do you prefer? (1/2): ").strip()
                if choice in ['1', '2']:
                    break
                print("Invalid choice. Please enter 1 or 2.")
            
            # Record preference
            chosen_idx = int(choice) - 1
            rejected_idx = 1 - chosen_idx
            
            preference_data.append({
                "prompt": prompt,
                "chosen": answers[chosen_idx],
                "rejected": answers[rejected_idx],
            })
            
            # Save after each new preference
            save_preference_data(preference_data, DATASET_PATH)
            
            # Clean up memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
    
    # 5. Prepare dataset for training
    dataset = prepare_datasets(preference_data)
    
    # Log memory before training
    log_memory_usage("Before training")
    
    # 6. Initialize trainer
    trainer = DPOTrainer(
        model=model,
        tokenizer=tokenizer,
        beta=BETA,
        learning_rate=LEARNING_RATE,
        num_epochs=NUM_EPOCHS,
        batch_size=BATCH_SIZE,
    )
    
    # 7. Train the model
    logger.info("Starting DPO training...")
    metrics = trainer.train(dataset)
    
    # 8. Save the final model
    trainer.save_model(ADAPTER_PATH)
    
    # Log memory after training
    log_memory_usage("After training")
    
    # 9. Evaluate the model
    logger.info("Evaluating the fine-tuned model...")
    test_questions = questions[:3]  # Use the first few questions for testing
    evaluation_results = evaluate_model(model, tokenizer, test_questions)
    
    # 10. Print results
    print("\n=== Model Evaluation Results ===")
    for question, answer in evaluation_results.items():
        print(f"Q: {question}")
        print(f"A: {answer}")
        print("-" * 40)
    
    # 11. Final report
    print("\n=== Training Complete ===")
    print(f"Processed {len(questions)} questions")
    print(f"Collected {len(preference_data)} preference pairs")
    print(f"Average DPO Loss: {metrics['loss']:.4f}")
    print(f"Average DPO Accuracy: {metrics['accuracy']:.4f}")
    print(f"Average Reward Gap: {metrics['reward_gap']:.4f}")
    print(f"Model adapter saved to: {ADAPTER_PATH}")
    
    # Final memory usage
    log_memory_usage("Final")

if __name__ == "__main__":
    main()