import os
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import Dataset
from tqdm import tqdm
from dotenv import load_dotenv
from deepeval.test_case import LLMTestCase

# Import our evaluator
from Code.Finetune.Reward_model.strategies.eval.g_eval.evaluator import GEvaluator

# Load environment variables from .env file
load_dotenv()

# Get API keys from environment variables
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
HF_READ_TOKEN = os.getenv('HF_READ_TOKEN')
HF_WRITE_TOKEN = os.getenv('HF_WRITE_TOKEN')

# Set environment variables
os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
os.environ["HF_READ_TOKEN"] = HF_READ_TOKEN
os.environ["HF_WRITE_TOKEN"] = HF_WRITE_TOKEN

class ScientificExplanationPipeline:
    def __init__(self, model_name="meta-llama/Llama-3.1-8B-Instruct"):
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.evaluator = GEvaluator()
        
    def load_model(self):
        """Load and prepare the model and tokenizer"""
        print("Loading model and tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            token=HF_READ_TOKEN
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            token=HF_READ_TOKEN,
            torch_dtype=torch.float16,
            device_map="auto"
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.model.config.pad_token_id = self.tokenizer.eos_token_id
            
    def prepare_training_data(self, data_path="Backend/Utils/training_data.json"):
        """Load and prepare training data"""
        print("Loading training data...")
        with open(data_path, 'r') as f:
            data = json.load(f)
            
        # Convert to format suitable for training
        prompts = []
        chosen_responses = []
        rejected_responses = []
        
        for item in data:
            prompts.append(item['prompt'])
            chosen_responses.append(item['chosen'])
            rejected_responses.append(item['rejected'])
            
        return Dataset.from_dict({
            'prompt': prompts,
            'chosen': chosen_responses,
            'rejected': rejected_responses
        })
        
    def generate_responses(self, questions):
        """Generate responses for evaluation"""
        print("Generating responses...")
        responses = []
        for question in tqdm(questions):
            inputs = self.tokenizer(question, return_tensors="pt").to(self.model.device)
            outputs = self.model.generate(
                **inputs,
                max_length=200,
                num_return_sequences=1,
                temperature=0.7
            )
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            responses.append(response)
        return responses
        
    def evaluate_responses(self, questions, responses):
        """Evaluate responses using GEvaluator"""
        print("Evaluating responses...")
        # Use the GEvaluator's batch evaluation
        evaluation_df = self.evaluator.evaluate_batch(questions, responses)
        
        # Convert DataFrame to our expected format
        results = {}
        for metric in self.evaluator.metrics:
            metric_name = metric.name
            results[metric_name] = {
                'scores': evaluation_df[metric_name].apply(lambda x: x['score']).tolist(),
                'reasons': evaluation_df[metric_name].apply(lambda x: x['reason']).tolist()
            }
            
        return results
        
    def generate_report(self, questions, responses, evaluation_results):
        """Generate a comprehensive evaluation report"""
        print("Generating evaluation report...")
        report = {
            'questions': questions,
            'responses': responses,
            'evaluation_results': evaluation_results
        }
        
        # Save report
        with open('evaluation_report.json', 'w') as f:
            json.dump(report, f, indent=4)
            
        # Create summary statistics
        summary = {}
        for metric, results in evaluation_results.items():
            scores = [s for s in results['scores'] if s is not None]
            if scores:
                summary[metric] = {
                    'mean': sum(scores) / len(scores),
                    'min': min(scores),
                    'max': max(scores)
                }
                
        with open('evaluation_summary.json', 'w') as f:
            json.dump(summary, f, indent=4)
            
        return report, summary

def main():
    # Initialize pipeline
    pipeline = ScientificExplanationPipeline()
    
    # Load model
    pipeline.load_model()
    
    # Load questions for evaluation
    with open('Backend/Utils/scientific_questions.txt', 'r') as f:
        questions = [line.strip() for line in f if line.strip()]
    
    # Generate responses
    responses = pipeline.generate_responses(questions)
    
    # Evaluate responses
    evaluation_results = pipeline.evaluate_responses(questions, responses)
    
    # Generate report
    report, summary = pipeline.generate_report(questions, responses, evaluation_results)
    
    print("Pipeline completed successfully!")
    print("\nEvaluation Summary:")
    for metric, stats in summary.items():
        print(f"\n{metric}:")
        print(f"  Mean: {stats['mean']:.2f}")
        print(f"  Min: {stats['min']:.2f}")
        print(f"  Max: {stats['max']:.2f}")

if __name__ == "__main__":
    main() 