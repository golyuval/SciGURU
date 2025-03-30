# Direct Preference Optimization (DPO) for Llama 3.1

This repository contains an implementation of Direct Preference Optimization (DPO) for fine-tuning Llama 3.1 models based on human preferences.

## Overview

DPO is a method for aligning language models with human preferences without the need to train a separate reward model. It directly optimizes a policy to maximize human preferences by minimizing a specific objective function.

### Key Features

- Interactive preference collection from users
- Fine-tuning using the DPO algorithm with PEFT (Parameter-Efficient Fine-Tuning)
- LoRA adapter-based fine-tuning for memory efficiency
- Persistent storage of preference data and model weights
- Testing framework to compare model outputs before and after fine-tuning

## Mathematical Background

DPO optimizes the following objective:

$\mathcal{L}_{\text{DPO}}(\pi_\theta; \pi_{\text{ref}}) = -\mathbb{E}_{(x,y_w,y_l) \sim \mathcal{D}} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$

Where:
- $\pi_\theta$ is the policy model being trained
- $\pi_{\text{ref}}$ is the reference model (frozen copy of the base model)
- $y_w$ is the preferred response
- $y_l$ is the less preferred response
- $\beta$ is a temperature parameter (controls regularization strength)
- $\sigma$ is the sigmoid function

This objective encourages the model to assign higher probability to preferred responses while staying close to the reference model.

## Installation

```bash
# Clone the repository
git clone your-repository-url
cd your-repository

# Install required packages
pip install transformers torch peft datasets tqdm
```

## Usage

### Running the DPO Fine-tuning

```bash
python backend/code/Finetune/Reward_model/strategies/dpo.py
```

The script will:
1. Load scientific questions from `backend/Utils/scientific_questions.txt`
2. Generate pairs of answers for each question
3. Ask you to select your preferred answer
4. Fine-tune the model using DPO based on your preferences
5. Save the adapter weights for future use

### Testing the Fine-tuned Model

```bash
python backend/code/Finetune/Reward_model/strategies/test_dpo.py --num_questions 3
```

This will:
1. Load the base model and the fine-tuned adapter
2. Generate answers to test questions using both models
3. Display a comparison of the responses

## File Structure

- `dpo.py`: Main implementation of the DPO algorithm
- `test_dpo.py`: Script to compare model outputs before and after fine-tuning
- `backend/Utils/scientific_questions.txt`: Source of scientific questions
- `backend/code/Finetune/Reward_model/strategies/dpo_adapter/`: Directory where the fine-tuned adapter is saved
- `backend/code/Finetune/Reward_model/strategies/preference_data.json`: Stored preference data

## How It Works

### 1. Preference Collection

The script generates two different answers for each scientific question and asks you to select which answer you prefer. This creates a dataset of preference pairs.

### 2. DPO Loss Calculation

For each preference pair, the DPO loss is calculated as:

```python
# Calculate the log likelihood ratio for chosen and rejected completions
chosen_rewards = policy_chosen_logps - reference_chosen_logps
rejected_rewards = policy_rejected_logps - reference_rejected_logps

# Calculate the DPO loss
logits = beta * (chosen_rewards - rejected_rewards)
losses = -torch.nn.functional.logsigmoid(logits)
```

This implements the mathematical formula described above.

### 3. Fine-tuning with PEFT

Instead of updating all model parameters, we use LoRA adapters to efficiently fine-tune the model:

```python
lora_config = LoraConfig(
    r=16,          # Rank of the update matrices
    lora_alpha=32, # Scaling factor
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
)
```

This allows fine-tuning an 8B parameter model on consumer hardware.

### 4. Persistent Adapter Weights

The fine-tuned adapter weights are saved to disk and automatically loaded in future sessions:

```python
# Check if adapter exists and load it
if os.path.exists(ADAPTER_PATH):
    logger.info(f"Loading existing adapter from {ADAPTER_PATH}")
    model = PeftModel.from_pretrained(model, ADAPTER_PATH)
else:
    # Add LoRA adapter
    logger.info("Adding new LoRA adapter to the model")
    model = get_peft_model(model, lora_config)
```

## Customization

You can adjust the following parameters in the script:

- `BETA`: Controls the regularization strength (default: 0.1)
- `LEARNING_RATE`: Learning rate for optimization (default: 5e-5)
- `NUM_EPOCHS`: Number of training epochs (default: 3)
- `BATCH_SIZE`: Batch size for training (default: 4)

LoRA configuration can also be modified:
- `r`: Rank of the update matrices
- `lora_alpha`: Scaling factor
- `target_modules`: Which modules to apply LoRA to

## References

- DPO Paper: [Direct Preference Optimization: Your Language Model is Secretly a Reward Model](https://arxiv.org/abs/2305.18290)
- LoRA Paper: [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)