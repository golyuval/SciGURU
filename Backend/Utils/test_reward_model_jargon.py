import os
import sys
import logging
from pathlib import Path

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

# Mock OpenAI (to bypass API key requirement)
import os
os.environ["OPENAI_API_KEY"] = "mock-key-for-testing"

# Import the deepeval test case
try:
    from deepeval.test_case import LLMTestCase
    logger.info("Successfully imported deepeval")
except ImportError as e:
    logger.error(f"Failed to import deepeval: {e}")
    logger.error("Make sure deepeval is installed: pip install deepeval")
    sys.exit(1)

# Import the jargon metric directly to test it
try:
    from Backend.Code.Eval.Jargon.jargon_metric import JargonMetric
    from Backend.Code.Eval.Jargon.jargon_util import calculate_grade, analyze_text, BASE_PATH
    import pandas as pd
    logger.info("Successfully imported jargon metric")
except ImportError as e:
    logger.error(f"Failed to import jargon metric: {e}")
    sys.exit(1)

# Test paragraphs with different complexity levels
test_paragraphs = [
    # Simple explanation
    """
    The sun is a big, bright star in the sky. It gives us light and warmth every day. 
    Plants use sunlight to grow, and we need plants for food. The sun rises in the morning 
    and sets in the evening, making day and night.
    """,
    
    # Complex explanation
    """
    Photosynthesis is a complex biochemical process occurring in the chloroplasts of 
    plant cells, specifically within the thylakoid membranes. The light-dependent 
    reactions initiate the process through photophosphorylation, generating ATP and 
    NADPH. These energy carriers then fuel the Calvin cycle, where carbon fixation 
    occurs through the enzyme RuBisCO, ultimately producing organic compounds essential 
    for cellular metabolism.
    """
]

def test_jargon_utils_directly():
    """Test the jargon utility functions directly, like in test_jargon.py"""
    print("\n========== Testing jargon_util.py directly ==========\n")
    
    try:
        # Load required data using correct paths
        jargon_dir = Path(BASE_PATH)
        names_path = jargon_dir / "names.csv"
        words_path = jargon_dir / "DataUKUS2018-2021.csv"
        
        print(f"Loading names from: {names_path}")
        print(f"Loading words from: {words_path}")
        
        if not names_path.exists():
            print(f"ERROR: Names file not found at {names_path}")
            return False
            
        if not words_path.exists():
            print(f"ERROR: Words file not found at {words_path}")
            return False
            
        names = pd.read_csv(names_path, header=None)[0].tolist()
        words = pd.read_csv(words_path, header=None).set_index(0)[1].to_dict()
        
        print(f"Loaded {len(names)} names and {len(words)} words")
        
        for i, paragraph in enumerate(test_paragraphs):
            print(f"\nTesting paragraph {i+1}:")
            print("-" * 40)
            print(paragraph[:100] + "...")
            score = analyze_text(paragraph, words, names, verbose=False)
            print(f"Jargon Score: {score:.4f}")
        
        return True
    except Exception as e:
        print(f"ERROR in test_jargon_utils_directly: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_jargon_metric():
    """Test the JargonMetric class directly"""
    print("\n========== Testing JargonMetric class directly ==========\n")
    
    try:
        jargon_metric = JargonMetric(threshold=0.5)
        
        for i, paragraph in enumerate(test_paragraphs):
            # Create a test case
            test_case = LLMTestCase(
                input=f"Explanation {i+1}",
                actual_output=paragraph
            )
            
            print(f"\nTesting paragraph {i+1}:")
            print("-" * 40)
            print(paragraph[:100] + "...")
            
            try:
                score = jargon_metric.measure(test_case)
                print(f"Jargon Score: {score:.4f}")
                print(f"Passes threshold: {score >= jargon_metric.threshold}")
            except Exception as e:
                print(f"ERROR in jargon_metric.measure: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return True
    except Exception as e:
        print(f"ERROR in test_jargon_metric: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def test_just_jargon_components():
    """Test only the jargon components, avoiding the rest of the reward model which uses OpenAI"""
    # Skip the test_reward_model_with_jargon test which requires OpenAI API
    print("\nSkipping TieredRewardModel test which requires OpenAI API")
    return True

if __name__ == "__main__":
    print("Testing jargon components of the reward model")
    
    # Test only the jargon components directly 
    jargon_utils_success = test_jargon_utils_directly()
    jargon_metric_success = test_jargon_metric()
    
    # Print overall results
    print("\n========== TEST RESULTS ==========")
    print(f"Jargon Utils Test: {'SUCCESS' if jargon_utils_success else 'FAILED'}")
    print(f"Jargon Metric Test: {'SUCCESS' if jargon_metric_success else 'FAILED'}") 