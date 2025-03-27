# used

t5_small = "t5-small"
t5_large = "t5-large"
bloom_7b = "bigscience/bloom-7b1"

# Models for Non-Commercial Use Only (Prioritized for SFT + RL)

mistral_7b = "mistralai/Mistral-7B"         # High performance, efficient, excels in SFT and RL
baichuan_2_7b = "baichuan-inc/Baichuan2-7B" # Multilingual, solid for science QA SFT
llama_3_1_8b = "meta-llama/Llama-3.1-8B"    # Balanced performance, RL works well
solar_10_7b = "Upstage/SOLAR-10.7B"         # Strong pretraining, suited for chain-of-thought RL
sailor_7b = "SEA/Sailor-7B"                 # Niche language support, less suited for broad science QA
minigpt_4 = "Vision-CAIR/MiniGPT-4"         # Vision-text capabilities, overkill for text-only QA

# Models for Commercial and Non-Commercial Use (Prioritized for SFT + RL)

falcon_mamba_7b = "tiiuae/falcon-mamba-7b"  # Scales well, excels in scientific QA tasks
qwen_7b = "Qwen/Qwen-7B"                    # Versatile, good multitask performance after SFT
h2ogpt_7b = "h2oai/h2ogpt-7b"               # Open-source, flexible for fine-tuning workflows
gpt_j_6b = "EleutherAI/gpt-j-6b"            # Lightweight, adaptable for SFT and RL workflows
