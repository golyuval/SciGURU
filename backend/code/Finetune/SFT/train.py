# run_sft_lora_llama3.1_8b.py
import os
import torch
import bitsandbytes as bnb
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training

# -----------------------------------------------------------
# 1. Config
# -----------------------------------------------------------
# Using the Llama 3.1 Instruct 8B model (requires authentication) in 8-bit mode.
BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # Ensure this is the correct model identifier.
OUTPUT_DIR = "./lora-Llama-3.1-8B-Instruct-sft"
BATCH_SIZE = 2
GRAD_ACCUM_STEPS = 4
LR = 1e-4
NUM_EPOCHS = 1  # Increase for real training

os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------------------------------------
# 2. Load Dataset
# -----------------------------------------------------------
dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
train_dataset = dataset["train"]

# -----------------------------------------------------------
# 3. Tokenizer
# -----------------------------------------------------------
print("Loading tokenizer...")

# Replace "your_hf_token" with your actual Hugging Face access token.
huggingFace_read_tok = os.getenv("HF_TOKEN")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_auth_token=huggingFace_read_tok)

# Some Llama models do not set pad_token by default.
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

def tokenize_function(examples):
    return tokenizer(examples["text"], truncation=True, max_length=512)

train_dataset = train_dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=["text"]
)

# -----------------------------------------------------------
# 4. Data Collator
# -----------------------------------------------------------
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer, mlm=False
)

# -----------------------------------------------------------
# 5. Load Base Model in 8-bit
# -----------------------------------------------------------
print("Loading Llama 3.1 Instruct 8B model in 8-bit precision...")

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_NAME,
    load_in_8bit=True,         # Use 8-bit quantization for memory savings
    device_map="auto",         # Automatic GPU placement
    use_auth_token=huggingFace_read_tok
)

# Prepare model for k-bit training (necessary for LoRA with 8-bit)
model = prepare_model_for_kbit_training(model)

# -----------------------------------------------------------
# 6. LoRA Configuration
# -----------------------------------------------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

model = get_peft_model(model, lora_config)

# -----------------------------------------------------------
# 7. Training Arguments
# -----------------------------------------------------------
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM_STEPS,
    evaluation_strategy="no",
    save_strategy="epoch",
    logging_steps=10,
    learning_rate=LR,
    fp16=True,
    optim="adamw_torch",
    report_to="none"  # Options: 'tensorboard', 'wandb', etc.
)

# -----------------------------------------------------------
# 8. Trainer
# -----------------------------------------------------------
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator
)

# -----------------------------------------------------------
# 9. Train
# -----------------------------------------------------------
print("Starting LoRA fine-tuning on Llama 3.1 Instruct 8B model ...")
trainer.train()

# -----------------------------------------------------------
# 10. Save Adapter
# -----------------------------------------------------------
print(f"Saving LoRA adapter to {OUTPUT_DIR} ...")
model.save_pretrained(OUTPUT_DIR)
