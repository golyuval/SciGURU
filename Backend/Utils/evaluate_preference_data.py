import os
import sys
import json
import logging
import signal
import time
from pathlib import Path
from dotenv import load_dotenv

# Setup logging first
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    handlers=[logging.StreamHandler()])
logger = logging.getLogger(__name__)

# Ensure we can find our modules
current_dir = Path(__file__).resolve().parent
repo_root = current_dir.parent.parent
logger.info(f"Current directory: {current_dir}")
logger.info(f"Repository root: {repo_root}")

# Add necessary directories to the path
sys.path.insert(0, str(repo_root))

# Load environment variables from .env file
dotenv_path = repo_root / ".env"
if os.path.exists(dotenv_path):
    load_dotenv(dotenv_path)
    logger.info(f"Loaded environment variables from {dotenv_path}")
else:
    logger.warning(f"No .env file found at {dotenv_path}")

# Global variables for state management
EVALUATED_DATA = []
INTERRUPTED = False
BATCH_SIZE = 10  # Number of pairs to evaluate before saving
AUTOSAVE_INTERVAL = 300  # Save every 5 minutes

# Check for OpenAI API key
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    logger.error("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")
    sys.exit(1)
else:
    logger.info("OPENAI_API_KEY found in environment variables")
    # Make sure the API key is available to all relevant modules
    os.environ["OPENAI_API_KEY"] = api_key

try:
    from deepeval.test_case import LLMTestCase
    logger.info("Successfully imported deepeval")
except ImportError as e:
    logger.error(f"Failed to import deepeval: {e}")
    logger.error("Make sure deepeval is installed: pip install deepeval")
    sys.exit(1)

# Try importing our reward model
try:
    from Backend.Code.Train.RewardModel.tiered_reward_model import TieredRewardModel
    logger.info("Successfully imported TieredRewardModel")
except ImportError as e:
    logger.error(f"Failed to import TieredRewardModel: {e}")
    logger.error("Check if the import path is correct and the module exists")
    sys.exit(1)

def load_preference_data(file_path=None):
    """Load the preference data from JSON file."""
    if file_path is None:
        file_path = current_dir / "preference_data.json"
    
    logger.info(f"Attempting to load data from: {file_path}")
    
    try:
        if not os.path.exists(file_path):
            logger.error(f"File not found: {file_path}")
            return []
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            logger.info(f"Loaded {len(data)} preference pairs from {file_path}")
            return data
    except Exception as e:
        logger.error(f"Error loading preference data: {str(e)}")
        return []

def save_evaluated_data(data, file_path=None, is_partial=False):
    """Save the evaluated preference data to a JSON file."""
    if file_path is None:
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        suffix = "_partial" if is_partial else ""
        file_path = current_dir / f"evaluated_preference_data{suffix}_{timestamp}.json"
    
    # Always create a backup of partial saves
    if is_partial:
        backup_path = current_dir / "evaluated_preference_data_latest.json"
        try:
            with open(backup_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            logger.info(f"Created backup at {backup_path}")
        except Exception as e:
            logger.warning(f"Error creating backup: {str(e)}")
        
    logger.info(f"Saving {'partial' if is_partial else 'final'} data to: {file_path}")
    
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved evaluated preference data to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Error saving evaluated data: {str(e)}")
        return False

def signal_handler(sig, frame):
    """Handle keyboard interrupt (Ctrl+C)."""
    global INTERRUPTED
    
    if INTERRUPTED:  # If pressed twice, exit immediately
        logger.warning("Forced exit. Data may be lost.")
        sys.exit(1)
        
    logger.warning("Interrupted! Saving partial results before exiting...")
    INTERRUPTED = True
    
    # Save current results
    if EVALUATED_DATA:
        save_evaluated_data(EVALUATED_DATA, is_partial=True)
        logger.info("Partial results saved. Exiting gracefully.")
    else:
        logger.warning("No data to save.")
    
    sys.exit(0)

def extract_question(prompt):
    """Extract the user's question from the prompt."""
    try:
        # Try to parse the prompt to extract just the question
        prompt_parts = prompt.split("<|end_header_id|>")
        user_question = ""
        for i, part in enumerate(prompt_parts):
            if "<|start_header_id|>user" in part and i+1 < len(prompt_parts):
                user_question = prompt_parts[i+1].split("<|eot_id|>")[0].strip()
                break
        
        # If we couldn't parse it, just use the whole prompt
        if not user_question:
            user_question = prompt
        
        return user_question
    except Exception as e:
        logger.warning(f"Error extracting question from prompt: {str(e)}")
        return prompt

def format_combined_request(prompt, chosen_response, rejected_response):
    """Format both answers in a single request to save API calls."""
    system_message = "You are a helpful assistant that answers scientific questions accurately and concisely."
    question = extract_question(prompt)
    
    # Format the request with both answers in a single prompt
    combined_request = f"""{system_message}

prompt: {question}
answer1: {chosen_response}
answer2: {rejected_response}

instructions:
please rate each of the answers above according to the metrics defined below, each metric will have further, more specific instructions and explanations.

articulation: 
1. Determine whether regardless of accuracy, this explanation is well articulated.
2. Output a grade between 0 and 10 based on the above criterion.
3. A grade of 10 means the explanation is perfectly articulated, while a grade of 0 means it is terribly articulated.
4. Do not take any other factors into account when giving a grade - correctness or accuracy should not affect the grade.
"""
    
    return combined_request

def evaluate_batch(reward_model, batch, start_idx):
    """Evaluate a batch of preference pairs."""
    global EVALUATED_DATA
    
    results = []
    batch_flipped = 0
    
    for idx, pair in enumerate(batch):
        overall_idx = start_idx + idx
        # Print only the progress count to terminal
        print(f"Evaluating pair {overall_idx+1}/{total_pairs}", flush=True)
        
        try:
            prompt = pair["prompt"]
            chosen_response = pair["chosen"]
            rejected_response = pair["rejected"]
            
            # Create a combined request with both answers
            combined_input = format_combined_request(prompt, chosen_response, rejected_response)
            
            # Create test cases for both responses
            chosen_test_case = LLMTestCase(
                input=combined_input,
                actual_output=chosen_response,
                expected_output=""
            )
            
            rejected_test_case = LLMTestCase(
                input=combined_input,
                actual_output=rejected_response,
                expected_output=""
            )
            
            try:
                # Calculate scores for both responses
                chosen_score = reward_model.calculate_reward(chosen_test_case)
                rejected_score = reward_model.calculate_reward(rejected_test_case)
                
                # Get detailed scores for analysis
                chosen_details = reward_model.get_detailed_scores(chosen_test_case)
                rejected_details = reward_model.get_detailed_scores(rejected_test_case)
                
                # Determine if the original selection was correct
                original_correct = chosen_score >= rejected_score
                
                # Prepare the evaluated pair
                evaluated_pair = {
                    "prompt": prompt,
                    "chosen": chosen_response if original_correct else rejected_response,
                    "rejected": rejected_response if original_correct else chosen_response,
                    "isSame": original_correct,
                    "scores": {
                        "original_chosen_score": chosen_score,
                        "original_rejected_score": rejected_score,
                        "chosen_details": chosen_details,
                        "rejected_details": rejected_details
                    }
                }
                
                results.append(evaluated_pair)
                
                if not original_correct:
                    batch_flipped += 1
                    logger.info(f"Flipped pair {overall_idx+1} - Chosen: {chosen_score:.2f}, Rejected: {rejected_score:.2f}")
                else:
                    logger.info(f"Kept pair {overall_idx+1} - Chosen: {chosen_score:.2f}, Rejected: {rejected_score:.2f}")
                    
            except Exception as e:
                logger.error(f"Error evaluating responses: {str(e)}")
                # Add the original pair without evaluation
                results.append({
                    "prompt": prompt,
                    "chosen": chosen_response,
                    "rejected": rejected_response,
                    "isSame": True,
                    "error": str(e)
                })
        except Exception as e:
            logger.error(f"Error processing pair {overall_idx+1}: {str(e)}")
            # Continue with the next pair
            continue
    
    return results, batch_flipped

def evaluate_responses():
    """
    Evaluate all responses in the preference data using the TieredRewardModel.
    Creates a new file with updated chosen/rejected pairs based on reward scores.
    """
    global EVALUATED_DATA, INTERRUPTED, total_pairs
    
    # Register signal handler for Ctrl+C
    signal.signal(signal.SIGINT, signal_handler)
    
    # Load preference data
    preference_data = load_preference_data()
    if not preference_data:
        logger.error("No preference data loaded, cannot continue")
        return False
    
    # Initialize the reward model
    try:
        reward_model = TieredRewardModel()
        logger.info("Initialized TieredRewardModel")
    except Exception as e:
        logger.error(f"Error initializing TieredRewardModel: {str(e)}")
        return False
    
    # Keep count of changes
    total_pairs = len(preference_data)
    flipped_pairs = 0
    
    # Track time for autosave
    last_save_time = time.time()
    
    # Process in batches
    for batch_start in range(0, total_pairs, BATCH_SIZE):
        # Check if interrupted
        if INTERRUPTED:
            break
            
        batch_end = min(batch_start + BATCH_SIZE, total_pairs)
        current_batch = preference_data[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start//BATCH_SIZE + 1}/{(total_pairs+BATCH_SIZE-1)//BATCH_SIZE}")
        
        # Evaluate the batch
        batch_results, batch_flipped = evaluate_batch(reward_model, current_batch, batch_start)
        
        # Add results to global data
        EVALUATED_DATA.extend(batch_results)
        flipped_pairs += batch_flipped
        
        # Check if we should autosave
        current_time = time.time()
        if (current_time - last_save_time) >= AUTOSAVE_INTERVAL:
            logger.info("Performing autosave...")
            save_evaluated_data(EVALUATED_DATA, is_partial=True)
            last_save_time = current_time
    
    # Save the final evaluated data
    if EVALUATED_DATA:
        success = save_evaluated_data(EVALUATED_DATA)
        
        # Print summary
        if success:
            logger.info(f"Evaluation complete: {flipped_pairs}/{total_pairs} pairs flipped ({flipped_pairs/total_pairs*100:.2f}%)")
            return True
    
    logger.error("Failed to evaluate responses or save results")
    return False

def print_evaluation_summary(file_path=None):
    """Print a summary of the evaluation results."""
    try:
        # Load the evaluated data
        if file_path is None:
            # Find the most recent evaluated data file
            data_files = list(current_dir.glob("evaluated_preference_data*.json"))
            if not data_files:
                logger.error("No evaluated data files found")
                return
                
            # Sort by modification time, most recent first
            data_files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
            file_path = data_files[0]
            
        logger.info(f"Loading evaluation data from: {file_path}")
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        # Count flipped vs kept pairs
        total = len(data)
        flipped = sum(1 for item in data if not item.get("isSame", True))
        
        print("\n===== EVALUATION SUMMARY =====")
        print(f"Data file: {file_path}")
        print(f"Total preference pairs evaluated: {total}")
        print(f"Pairs where selection was kept: {total - flipped} ({(total-flipped)/total*100:.2f}%)")
        print(f"Pairs where selection was flipped: {flipped} ({flipped/total*100:.2f}%)")
        
        # Show some examples of flipped pairs
        if flipped > 0:
            print("\n----- EXAMPLE FLIPPED PAIRS -----")
            flipped_examples = [item for item in data if not item.get("isSame", True)]
            for i, example in enumerate(flipped_examples[:3]):  # Show up to 3 examples
                print(f"\nExample {i+1}:")
                print(f"Original chosen score: {example['scores']['original_chosen_score']:.2f}")
                print(f"Original rejected score: {example['scores']['original_rejected_score']:.2f}")
                print("Tier scores comparison:")
                for tier in ['tier1', 'tier2', 'tier3']:
                    orig_chosen = example['scores']['chosen_details']['tier_scores'][tier]['score']
                    orig_rejected = example['scores']['rejected_details']['tier_scores'][tier]['score']
                    print(f"  {tier}: Chosen: {orig_chosen:.2f}, Rejected: {orig_rejected:.2f}")
        
    except Exception as e:
        logger.error(f"Error loading evaluation summary: {str(e)}")

if __name__ == "__main__":
    logger.info("Starting evaluation of preference data...")
    success = evaluate_responses()
    
    if success:
        print_evaluation_summary()
    elif EVALUATED_DATA:
        # If we have partial results but didn't complete successfully
        logger.info("Evaluation didn't complete successfully, but partial results are available.")
        print_evaluation_summary()
    else:
        logger.error("Evaluation failed. Check logs for details.") 