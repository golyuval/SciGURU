# -*- coding: utf-8 -*-
"""
PPO Implementation for Scientific Explanation RLHF
-------------------------------------------------
This script implements PPO for fine-tuning LLMs to simplify scientific explanations.
The reward model integration is left for you to connect with your g-eval and jargonizer.
"""

import os
import torch
import warnings
import logging
import numpy as np
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    BitsAndBytesConfig, 
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead, create_reference_model
from typing import Dict, List, Optional, Tuple
import torch.nn as nn
from torch.optim import Adam
from .reward_model import ScientificExplanationRewardModel

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Path configurations - adjust these to match your project structure
SCIENTIFIC_QUESTIONS_PATH = "backend/Utils/scientific_questions.txt"
OUTPUT_DIR = "./output/llama3.1-sci-explain-ppo"
CHECKPOINT_DIR = f"{OUTPUT_DIR}/checkpoints"

# Create necessary directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# PPO Configuration
ppo_config = PPOConfig(
    learning_rate=1.41e-5,
    mini_batch_size=4,       # Adjust based on your GPU memory
    batch_size=16,           # Number of prompts to process before optimization step
    gradient_accumulation_steps=4, # Effective batch size = batch_size * gradient_accumulation_steps
    optimize_cuda_cache=True,
    target_kl=0.1,           # Target KL divergence for adaptive KL penalty
    kl_penalty='kl',         # Use KL penalty
    seed=42,
    use_score_scaling=True,  # Scale rewards using running statistics
    use_score_norm=True,     # Normalize rewards
    score_clip=None,         # Don't clip rewards
    remove_unused_columns=False, # Keep inputs needed by reward model
    log_with="tensorboard",  # Enable tensorboard logging
    logging_dir=f"{OUTPUT_DIR}/logs"
)

# LoRA Configuration for parameter-efficient fine-tuning
lora_config = LoraConfig(
    r=16,                     # Rank of the update matrices
    lora_alpha=32,            # Alpha parameter for scaling
    lora_dropout=0.05,        # Dropout probability
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)

# Quantization Config (for reduced memory usage)
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16, # Use bfloat16 for computation
    bnb_4bit_use_double_quant=True,
)

# Generation config for sampling responses
generation_kwargs = {
    "min_length": -1,        # Don't ignore the EOS token
    "top_k": 0.0,            # No top-k sampling
    "top_p": 0.9,            # Use nucleus sampling with 0.9 threshold
    "temperature": 0.7,      # Temperature for sampling
    "do_sample": True,       # Sample from the distribution
    "pad_token_id": None,    # Will be set after loading tokenizer
    "max_new_tokens": 512,   # Maximum length of generated response
    "eos_token_id": None,    # Will be set after loading tokenizer
}

def load_scientific_questions(file_path=SCIENTIFIC_QUESTIONS_PATH):
    """
    Load scientific questions from the specified text file.
    Each question should be in a separate line.
    """
    if not os.path.exists(file_path):
        logger.warning(f"Questions file not found at {file_path}. Using demo questions.")
        # Fallback demo questions
        return [
            "Explain quantum entanglement in simple terms.",
            "How does gene editing with CRISPR work?",
            "What is dark matter and why do scientists think it exists?",
            "Explain how mRNA vaccines work to a 10-year-old.",
            "What are black holes and how do they form?"
        ]
    
    questions = []
    with open(file_path, 'r') as f:
        for line in f:
            # Parse the line (format may vary, adjust as needed)
            line = line.strip()
            if line.startswith('-'):
                question = line[1:].strip()  # Remove the leading dash and whitespace
            else:
                question = line
            if question:  # Skip empty lines
                questions.append(question)
    
    logger.info(f"Loaded {len(questions)} questions from {file_path}")
    return questions

def format_prompt(question, system_prompt=None):
    """
    Format a question using Llama 3.1 Instruct chat template.
    Optionally add a system prompt to guide response quality.
    """
    if system_prompt is None:
        system_prompt = "You are a helpful assistant that explains complex scientific concepts in simple, easy-to-understand language. Use analogies, simple vocabulary, and short sentences."
    
    return f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>\n<|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|>\n<|start_header_id|>assistant<|end_header_id|>\n\n"

def prepare_dataset(questions):
    """
    Prepare a dataset of formatted prompts from scientific questions.
    """
    # Format all prompts
    formatted_prompts = [format_prompt(q) for q in questions]
    
    # Create a Dataset object
    dataset_dict = {'query': formatted_prompts, 'original_question': questions}
    dataset = Dataset.from_dict(dataset_dict)
    
    logger.info(f"Created dataset with {len(dataset)} formatted prompts")
    return dataset

def setup_models_and_tokenizer():
    """
    Set up the tokenizer, actor model, and reference model.
    """
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct", 
        trust_remote_code=True
    )
    
    # Ensure tokenizer has pad token
    if tokenizer.pad_token is None:
        logger.info("Setting EOS token as pad token")
        tokenizer.pad_token = tokenizer.eos_token
    
    tokenizer.padding_side = 'left'
    generation_kwargs["pad_token_id"] = tokenizer.pad_token_id
    generation_kwargs["eos_token_id"] = tokenizer.eos_token_id
    ppo_config.stop_token_id = tokenizer.eos_token_id
    
    logger.info("Loading actor model...")
    actor_model = AutoModelForCausalLMWithValueHead.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Resize token embeddings if pad token was added
    actor_model.pretrained_model.resize_token_embeddings(len(tokenizer))
    
    # Prepare actor model for training
    if hasattr(actor_model, "is_quantized") and actor_model.is_quantized:
        logger.info("Preparing quantized model for k-bit training...")
        actor_model = prepare_model_for_kbit_training(
            actor_model, 
            use_gradient_checkpointing=True
        )
    
    # Apply LoRA for parameter-efficient fine-tuning
    logger.info("Applying LoRA to the actor model...")
    actor_model = get_peft_model(actor_model, lora_config)
    actor_model.print_trainable_parameters()
    
    # Fix model attributes needed by the PPO trainer
    if not hasattr(actor_model, "generation_config"):
        actor_model.generation_config = actor_model.pretrained_model.generation_config
    if not hasattr(actor_model, "base_model_prefix"):
        actor_model.base_model_prefix = "pretrained_model"
    
    # Create reference model for KL divergence calculation
    logger.info("Creating reference model...")
    ref_model = create_reference_model(actor_model, num_shared_layers=None)
    
    return tokenizer, actor_model, ref_model

def tokenize_dataset(dataset, tokenizer):
    """
    Tokenize the dataset for the PPO Trainer.
    """
    def tokenize_fn(examples):
        tokenized_output = tokenizer(
            examples["query"],
            truncation=True,
            max_length=512,
            padding="max_length"
        )
        return tokenized_output
    
    dataset = dataset.map(tokenize_fn, batched=True)
    dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])
    return dataset

def save_checkpoint(trainer, epoch):
    """
    Save a checkpoint of the model.
    """
    checkpoint_path = f"{CHECKPOINT_DIR}/checkpoint-{epoch}"
    os.makedirs(checkpoint_path, exist_ok=True)
    
    logger.info(f"Saving checkpoint to {checkpoint_path}")
    trainer.save_model(checkpoint_path)
    
    # Save tokenizer alongside model
    trainer.tokenizer.save_pretrained(checkpoint_path)

# ------------------------------
# This is where you'll plug in your custom reward model
# ------------------------------
def compute_rewards(model_responses, questions, trainer=None):
    """
    Placeholder for your custom reward model using g-eval and jargonizer.
    
    You'll need to implement this function to:
    1. Process the model's responses
    2. Use your g-eval model to assess explanation quality
    3. Use your jargonizer to measure simplicity
    4. Combine these signals into a reward score
    
    Args:
        model_responses (list): List of text responses from the model
        questions (list): Original scientific questions
        trainer: The PPO trainer instance, for accessing device information
        
    Returns:
        list: Reward scores for each response (torch tensors on the correct device)
    """
    # REPLACE THIS with your actual reward computation
    # This is just a placeholder - DO NOT USE THIS IMPLEMENTATION
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if trainer is not None:
        device = trainer.accelerator.device
    
    # Placeholder rewards - you need to replace this with actual g-eval and jargonizer logic
    placeholder_rewards = [torch.tensor(0.5, device=device) for _ in model_responses]
    
    # Log a warning to remind about implementation
    logger.warning("Using placeholder rewards - implement your g-eval and jargonizer here!")
    
    # Example structure of how you might implement the real reward function:
    """
    rewards = []
    
    for response, question in zip(model_responses, questions):
        # 1. Evaluate explanation quality using g-eval
        quality_score = your_geval_model.evaluate(question, response)
        
        # 2. Evaluate simplicity using jargonizer (lower complexity is better)
        complexity_score = your_jargonizer.measure_complexity(response)
        simplicity_score = 1.0 - (complexity_score / 100.0)  # Normalize to [0,1]
        
        # 3. Combine the scores (adjust weights as needed)
        combined_reward = 0.7 * quality_score + 0.3 * simplicity_score
        
        # 4. Convert to tensor and add to rewards list
        reward_tensor = torch.tensor(combined_reward, device=device)
        rewards.append(reward_tensor)
    
    return rewards
    """
    
    return placeholder_rewards

# ------------------------------
# PPO Training Loop
# ------------------------------
def train():
    """
    Main training function.
    """
    # Load tokenizer and models
    tokenizer, actor_model, ref_model = setup_models_and_tokenizer()
    
    # Load scientific questions
    questions = load_scientific_questions()
    
    # Prepare dataset
    dataset = prepare_dataset(questions)
    tokenized_dataset = tokenize_dataset(dataset, tokenizer)
    
    # Data collator for padding
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    
    # Configure model settings for PPO trainer
    for model in [actor_model, ref_model]:
        # Ensure return_dict is set correctly within the model structure
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            model.base_model.model.config.return_dict = True
            if hasattr(model.base_model.model, "pretrained_model"):
                model.base_model.model.pretrained_model.config.return_dict = True
    
    # Initialize PPO trainer
    logger.info("Initializing PPO trainer...")
    ppo_trainer = PPOTrainer(
        args=ppo_config,
        model=actor_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    # Training loop
    logger.info("Starting PPO training...")
    num_epochs = 3  # Set the number of training epochs
    
    for epoch in range(num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{num_epochs}")
        
        # Get batches from the dataset
        for batch_idx, batch in enumerate(ppo_trainer.dataloader):
            # Move batch to device
            batch = {k: v.to(ppo_trainer.accelerator.device) if isinstance(v, torch.Tensor) else v 
                     for k, v in batch.items()}
            
            # Extract query tensors (input_ids) from the batch
            query_tensors = batch["input_ids"]
            
            # Get the original questions for this batch
            # This assumes dataset ordering is maintained - you may need to add indices to the dataset
            batch_indices = list(range(batch_idx * ppo_trainer.config.batch_size, 
                                     min((batch_idx + 1) * ppo_trainer.config.batch_size, len(questions))))
            batch_questions = [questions[i % len(questions)] for i in batch_indices]
            
            # Generate responses with the current model
            logger.info(f"Generating responses for batch {batch_idx+1}...")
            response_tensors = ppo_trainer.generate(
                query_tensors,
                return_prompt=False,
                **generation_kwargs
            )
            
            # Decode the responses
            batch_responses = [
                tokenizer.decode(r.squeeze(), skip_special_tokens=True)
                for r in response_tensors
            ]
            
            # Compute rewards using your custom reward model
            logger.info(f"Computing rewards for batch {batch_idx+1}...")
            rewards = compute_rewards(batch_responses, batch_questions, ppo_trainer)
            
            # Log a sample of responses and rewards
            for i in range(min(2, len(batch_responses))):
                logger.info(f"Sample {i+1}:")
                logger.info(f"Question: {batch_questions[i]}")
                logger.info(f"Response: {batch_responses[i][:100]}...")
                logger.info(f"Reward: {rewards[i].item()}")
            
            # Run PPO step
            logger.info(f"Running PPO optimization for batch {batch_idx+1}...")
            stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
            
            # Log statistics
            logger.info(f"Epoch {epoch+1}, Batch {batch_idx+1}: "
                        f"Mean reward: {stats['ppo/mean_scores']:.4f}, "
                        f"Loss: {stats.get('ppo/loss/total', 'N/A')}, "
                        f"KL: {stats.get('ppo/kl', 'N/A')}")
            
        # Save checkpoint after each epoch
        save_checkpoint(ppo_trainer, epoch+1)
    
    # Save the final model
    logger.info("Saving final model...")
    ppo_trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    logger.info("Training completed successfully!")

def evaluate_model(model_path=OUTPUT_DIR):
    """
    Evaluate the fine-tuned model on a few sample questions.
    """
    logger.info(f"Evaluating model from {model_path}...")
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    
    # Load model
    from peft import PeftModel
    from transformers import AutoModelForCausalLM
    
    base_model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Meta-Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    model = PeftModel.from_pretrained(base_model, model_path)
    model.eval()
    
    # Test questions
    test_questions = [
        "Explain how DNA replication works in simple terms.",
        "What is quantum computing and how is it different from regular computing?",
        "Explain the greenhouse effect in a way a middle school student would understand.",
    ]
    
    for question in test_questions:
        # Format prompt
        formatted_prompt = format_prompt(question)
        inputs = tokenizer(formatted_prompt, return_tensors="pt").to(model.device)
        
        # Generate response
        logger.info(f"Generating response for: {question}")
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        
        # Decode and print response
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        logger.info(f"Response: {response}")
        logger.info("-" * 50)

class PPOTrainer:
    def __init__(
        self,
        model_name: str,
        reward_model: Optional[ScientificExplanationRewardModel] = None,
        learning_rate: float = 1e-5,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        self.device = device
        self.model = AutoModelForCausalLM.from_pretrained(model_name).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.reward_model = reward_model or ScientificExplanationRewardModel()
        
        # Initialize optimizers
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        
        # PPO hyperparameters
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        
    def generate_response(self, prompt: str, max_length: int = 200) -> str:
        """Generate a response using the current policy."""
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.model.generate(
            **inputs,
            max_length=max_length,
            num_return_sequences=1,
            do_sample=True,
            temperature=0.7,
            pad_token_id=self.tokenizer.eos_token_id
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def compute_reward(
        self,
        question: str,
        answer: str,
        expected_answer: Optional[str] = None
    ) -> float:
        """Compute reward using our scientific explanation reward model."""
        return self.reward_model.calculate_reward(question, answer, expected_answer)
    
    def compute_advantages(
        self,
        rewards: torch.Tensor,
        values: torch.Tensor,
        dones: torch.Tensor
    ) -> torch.Tensor:
        """Compute advantages using GAE."""
        advantages = torch.zeros_like(rewards)
        last_gae = 0
        
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
            else:
                next_value = values[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gamma * last_gae * (1 - dones[t])
            advantages[t] = last_gae
        
        return advantages
    
    def train_step(
        self,
        questions: List[str],
        answers: List[str],
        expected_answers: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Perform one training step of PPO."""
        # Compute rewards
        rewards = torch.tensor([
            self.compute_reward(q, a, e)
            for q, a, e in zip(questions, answers, expected_answers or [None] * len(questions))
        ]).to(self.device)
        
        # Get old policy logits
        with torch.no_grad():
            old_logits = []
            for answer in answers:
                inputs = self.tokenizer(answer, return_tensors="pt").to(self.device)
                outputs = self.model(**inputs)
                old_logits.append(outputs.logits)
        
        # Compute advantages
        advantages = self.compute_advantages(
            rewards,
            torch.zeros_like(rewards),  # Placeholder for values
            torch.zeros_like(rewards)   # Placeholder for dones
        )
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        # Compute policy loss
        policy_loss = 0
        value_loss = 0
        entropy_loss = 0
        
        for i, answer in enumerate(answers):
            inputs = self.tokenizer(answer, return_tensors="pt").to(self.device)
            outputs = self.model(**inputs)
            
            # Compute ratio
            ratio = torch.exp(outputs.logits - old_logits[i])
            
            # Compute surrogate loss
            surr1 = ratio * advantages[i]
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages[i]
            policy_loss += -torch.min(surr1, surr2).mean()
            
            # Compute value loss
            value_loss += nn.MSELoss()(outputs.logits.mean(), rewards[i])
            
            # Compute entropy loss
            entropy_loss += -torch.mean(torch.exp(outputs.logits) * outputs.logits)
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss + self.entropy_coef * entropy_loss
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy_loss": entropy_loss.item(),
            "total_loss": loss.item()
        }
    
    def train(
        self,
        questions: List[str],
        expected_answers: List[str],
        num_epochs: int = 10,
        batch_size: int = 8
    ):
        """Train the model using PPO."""
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch + 1}/{num_epochs}")
            
            # Generate responses
            answers = [self.generate_response(q) for q in questions]
            
            # Train on batches
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i + batch_size]
                batch_answers = answers[i:i + batch_size]
                batch_expected = expected_answers[i:i + batch_size]
                
                metrics = self.train_step(batch_questions, batch_answers, batch_expected)
                
                print(f"Batch {i//batch_size + 1}:")
                for metric, value in metrics.items():
                    print(f"  {metric}: {value:.4f}")
            
            # Save checkpoint
            self.save_checkpoint(f"checkpoint_epoch_{epoch + 1}.pt")
    
    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

if __name__ == "__main__":
    # Try to authenticate to Hugging Face Hub if needed
    try:
        from huggingface_hub import login
        login(token=os.environ.get("HF_TOKEN"))
    except:
        logger.warning("Could not authenticate to Hugging Face Hub. This is only necessary if using gated models.")
    
    # Start training
    train()
    
    # Evaluate the model
    evaluate_model()