from dotenv import load_dotenv
from pydantic import BaseModel, Field
import os
from dotenv import load_dotenv

# Load the variables from .env file
load_dotenv()

class sft_config(BaseModel):

    # General
    out_dir             : str       = Field(default= "./versions/SciGURU_alpha"             , description="Output directory to save training objects")
    base_model          : str       = Field(default= "meta-llama/Llama-3.1-8B-Instruct"     , description="Initial foundational LLM")
    API_openai          : str       = Field(default= os.environ.get("API_openai")           , description="OpenAI API")
    API_hugging_face    : str       = Field(default= os.environ.get("API_hugging_face")     , description="Hugging Face API")

    # Training parameters
    batch_size          : int       = Field(default= 2      , description="Batch size for training")
    grad_accum_steps    : int       = Field(default= 4      , description="Number of gradient accumulation steps")
    learning_rate       : float     = Field(default= 1e-4   , description="Learning rate for training")
    num_epochs          : int       = Field(default= 1      , description="Number of training epochs")
    
    # LoRA parameters
    lora_r              : int       = Field(default= 8      , description="Low-rank dimension for LoRA")
    lora_alpha          : int       = Field(default= 32     , description="Scaling factor for LoRA")
    lora_dropout        : float     = Field(default= 0.05   , description="Dropout rate for LoRA layers")
    lora_target         : list      = Field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"], description="Target modules for applying LoRA")
    
    # Quantization and model loading
    load_in_8bit        : bool      = Field(default= True   , description="Load the model in 8-bit precision")
    device_map          : str       = Field(default= "auto" , description="Device mapping for model loading")



import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


BASE_MODEL_NAME = "meta-llama/Llama-3.1-8B-Instruct"  # base model

AUTH_TOKEN = "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" # huggingface

ADAPTER_DIR = "./lora-Llama-3.1-8B-Instruct-sft" # finetuned


# # ---------- Hugging Face : pre trained  ---------------------------------------------------------------------

# print("Loading tokenizer...")
# tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, use_auth_token=AUTH_TOKEN)

# print("Loading base model in 8-bit precision...")
# base_model = AutoModelForCausalLM.from_pretrained(
#     BASE_MODEL_NAME,
#     load_in_8bit=True,         # 8-bit quantization for memory savings.
#     device_map="auto",         # Automatic device placement.
#     use_auth_token=AUTH_TOKEN
# )

# # After loading your model:
# base_model.save_pretrained("./local_model")
# tokenizer.save_pretrained("./local_tokenizer")

# ---------- Local : pre trained ---------------------------------------------------------------------


# And later, you can load it from the local directory:
base_model = AutoModelForCausalLM.from_pretrained(
    "./local_model",
    load_in_8bit=True,
    device_map="auto"
)

tokenizer = AutoTokenizer.from_pretrained("./local_tokenizer")


# ---------- Local : Trained ---------------------------------------------------------------------


print("Loading fine-tuned model with LoRA adapter...")
# Wrap the base model with the LoRA adapter.
trained_model = PeftModel.from_pretrained(base_model, ADAPTER_DIR)


# ---------- Inference ---------------------------------------------------------------------


def generate_answer(model, prompt, max_new_tokens=150):

    # Tokenize the prompt and move to the same device as the model.
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    
    # Generate output (using sampling; adjust parameters as needed).
    output_ids = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=True,
        temperature=0.7
    )
    # Decode generated tokens and skip special tokens.
    output_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return output_text


questions = [
    "What is quantum tunneling and how does it work?",
    "Explain the concept of entropy in thermodynamics.",
    "How do gravitational waves work?"
]


for q in questions:
    
    # Generate and print response from the base model.
    print("\nResponse:")
    base_response = generate_answer(base_model, q)
    print(base_response + "\n")
    

