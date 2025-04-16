# Scientific Explanation Pipeline

This project implements a pipeline for fine-tuning and evaluating LLMs to generate better scientific explanations. The pipeline uses Direct Preference Optimization (DPO) for fine-tuning and G-Eval for comprehensive evaluation of the model's responses.

## Pipeline Overview

The pipeline consists of the following main components:

pip install -r requirements.txt

1. **Model Loading and Preparation**
   - Loads the base LLM (default: Llama-3-8b)
   - Prepares the tokenizer and model configuration
   - Sets up necessary padding tokens

2. **Training Data Preparation**
   - Loads training data from `training_data.json`
   - Formats data for DPO training
   - Creates a dataset with prompts, chosen responses, and rejected responses

3. **Model Fine-tuning**
   - Implements DPO training
   - Uses Hugging Face's Trainer class
   - Configures training arguments for optimal performance

4. **Response Generation**
   - Generates responses for scientific questions
   - Uses temperature sampling for diverse outputs
   - Handles tokenization and decoding

5. **G-Eval Evaluation**
   - Implements comprehensive evaluation using multiple metrics:
     - Zemla Metrics (internal coherence, completeness, alternatives, articulation, perceived truth)
     - Explanation Quality Metrics (explanation type, correctness, metaphor, content units, connection to everyday life, humor, analogy)
     - Readability Metrics (Flesch-Kincaid, Flesch Reading Ease, Dale-Chall, ARI)
     - Jargon Metric

6. **Report Generation**
   - Creates detailed evaluation reports
   - Generates summary statistics
   - Saves results in JSON format

## Environment Setup

1. Create a `.env` file in the project root with the following structure:
```bash
# OpenAI API Key
OPENAI_API_KEY=your_openai_api_key

# Hugging Face Tokens
HF_READ_TOKEN=your_hf_read_token
HF_WRITE_TOKEN=your_hf_write_token
```

2. Install additional dependencies:
```bash
pip install python-dotenv
```

3. Ensure all other dependencies are installed:
```bash
pip install transformers datasets torch pandas tqdm
```

4. The environment variables will be automatically loaded when running the pipeline.

## Usage

1. Ensure all dependencies are installed:
   ```bash
   pip install transformers datasets torch pandas tqdm
   ```

2. Set up the OpenAI API key in the environment:
   ```python
   OPENAI_API_KEY = "your-api-key"
   os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
   ```

3. Run the pipeline:
   ```bash
   python run.py
   ```

## Output Files

The pipeline generates two main output files:

1. `evaluation_report.json`: Contains detailed evaluation results for each response
2. `evaluation_summary.json`: Provides summary statistics for each metric

## G-Eval Implementation

G-Eval uses an outer LLM (GPT) to evaluate responses based on specific criteria:

1. **Evaluation Process**:
   - Takes a response to evaluate
   - Provides specific evaluation criteria and steps
   - Uses GPT to analyze the response
   - Returns a score and explanation

2. **Metrics Implementation**:
   - Each metric is implemented as a separate class
   - Metrics use LLMTestCase for standardized evaluation
   - Results include both scores and explanations

3. **Error Handling**:
   - Gracefully handles evaluation failures
   - Provides detailed error messages
   - Continues evaluation even if some metrics fail

## Training Data Format

The training data should be in JSON format with the following structure:
```json
[
    {
        "prompt": "scientific question",
        "chosen": "preferred response",
        "rejected": "less preferred response"
    }
]
```

## Configuration

Key configuration parameters can be adjusted in the `ScientificExplanationPipeline` class:

- Model name and size
- Training parameters (epochs, batch size, learning rate)
- Generation parameters (max length, temperature)
- Evaluation metrics

## Dependencies

- transformers
- datasets
- torch
- pandas
- tqdm
- OpenAI API
- Custom metrics from g_eval, readability_metrics, and Jargon packages

# Developers Guide ( SLURM )

</br>


## Clone SciGURU repository

### 1 - Enter root directory

- via MobaXtrem - open terminal
- via SSH session </br>

    ```bash
    ssh <username>:slurm.bgu.ac.il
    <password>
    ```

### 2 - Clone repository
```bash
git clone https://github.com/golyuval/SciGURU
username : golyuval
password : <repository access token>
```

</br>

--- 



## Reset Environment



### 1 - Enter SciGURU directory
```bash
cd SciGURU/
```

### 2 - Install packages
```bash
./Backend/Scripts/reset_env.sh
```

### 3 - Activate my_env (if not activated)
```bash
conda activate my_env
```

### 4 - Save critical tokens
```bash
export HF_READ_TOKEN=<hugging_face_read_token>
export HF_WRITE_TOKEN=<hugging_face_write_token>
```

--- 



## Run Scripts


### 1 - Enter backend directory
```bash
cd SciGURU/Backend/
```

### 2 - Run Scripts

**Load Data** - task for loading data from hugging face into **Code/Train/SFT/datasets**
```bash
sbatch Scripts/Train/load_data.sbatch
```

**SFT** - task for performing SFT training (rtx_4090) ---> save model to **Versions** 
```bash
sbatch Scripts/Train/SFT.sbatch
```

**SFT_** - same task as SFT (rtx_3090) 
```bash
sbatch Scripts/Train/SFT_.sbatch
```
</br>

