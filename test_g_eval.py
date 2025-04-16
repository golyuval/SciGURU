import os
import sys
from dotenv import load_dotenv
from openai import OpenAI
from deepeval.test_case import LLMTestCase

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in environment variables. Please set it in your .env file.")

# Set the API key in the environment
os.environ["OPENAI_API_KEY"] = api_key

from Backend.Code.Finetune.Reward_model.strategies.eval.g_eval import (
    internal_coherence_metric,
    completeness_metric,
    alternatives_metric,
    articulation_metric,
    perceived_truth_metric,
    explanation_type_metric,
    correctness_metric,
    metaphor_metric,
    content_units_metric,
    connection_to_everyday_life_metric,
    analogy_metric
)

def test_g_eval_metrics():
    try:
        # Test cases
        test_cases = [
            {
                "question": "What is photosynthesis?",
                "answer": "Photosynthesis is the process by which plants convert sunlight into energy. They use chlorophyll to capture light and convert carbon dioxide and water into glucose and oxygen.",
                "expected_answer": "Photosynthesis is the biological process where plants convert light energy from the sun into chemical energy (glucose). During this process, plants use chlorophyll (a green pigment) to absorb sunlight, take in carbon dioxide from the air through their leaves, and water from the soil through their roots. The end products are glucose (food for the plant) and oxygen, which is released into the atmosphere.",
                "description": "Basic scientific explanation"
            },
            {
                "question": "How does quantum entanglement work?",
                "answer": "Quantum entanglement is a phenomenon where two or more particles become correlated in such a way that the state of one particle instantly influences the state of the other, regardless of distance.",
                "expected_answer": "Quantum entanglement occurs when two or more particles are generated, interact, or share spatial proximity in a way such that the quantum state of each particle cannot be described independently. Instead, a quantum mechanical description of the system can only be given for the pair or group as a whole. When particles are entangled, measuring the state of one particle immediately determines the state of the other particle(s), regardless of the distance between them. This connection appears to occur instantaneously, which Einstein famously called 'spooky action at a distance.'",
                "description": "Complex scientific concept"
            },
            {
                "question": "Explain the concept of gravity",
                "answer": "Gravity is like a giant invisible rubber sheet. When you place objects on it, they create dents. The bigger the object, the bigger the dent. Other objects roll towards these dents, which is why things fall towards the Earth.",
                "expected_answer": "Gravity is a fundamental force of nature that attracts any objects with mass or energy toward each other. According to Einstein's theory of general relativity, gravity is not a force but a consequence of the curvature of spacetime caused by mass and energy. The more massive an object is, the more it warps the fabric of spacetime around it, causing other objects to follow curved paths in its vicinity. This is often visualized using the analogy of a heavy ball placed on a rubber sheet, creating a depression that causes smaller objects to roll toward it.",
                "description": "Explanation with analogy"
            }
        ]
        
        print("\n🔍 Testing G-Eval Metrics\n")
        print("=" * 80)
        
        for test_case in test_cases:
            print(f"\nTesting: {test_case['description']}")
            print("-" * 40)
            print(f"Question: {test_case['question']}")
            print(f"Answer: {test_case['answer']}")
            print(f"Expected Answer: {test_case['expected_answer']}")
            
            # Create test case with expected output
            test_case_obj = LLMTestCase(
                input=test_case['question'],
                actual_output=test_case['answer'],
                expected_output=test_case['expected_answer']
            )
            
            # Test Zemla metrics
            print("\nZemla Metrics:")
            metrics = [
                ("Internal Coherence", internal_coherence_metric),
                ("Completeness", completeness_metric),
                ("Alternatives", alternatives_metric),
                ("Articulation", articulation_metric),
                ("Perceived Truth", perceived_truth_metric)
            ]
            
            for metric_name, metric in metrics:
                try:
                    metric.measure(test_case_obj)
                    print(f"{metric_name}: {metric.score:.2f}")
                except Exception as e:
                    print(f"{metric_name} failed: {str(e)}")
            
            # Test Explanation Quality metrics
            print("\nExplanation Quality Metrics:")
            quality_metrics = [
                ("Explanation Type", explanation_type_metric),
                ("Correctness", correctness_metric),
                ("Metaphor", metaphor_metric),
                ("Content Units", content_units_metric),
                ("Connection to Everyday Life", connection_to_everyday_life_metric),
                ("Analogy", analogy_metric)
            ]
            
            for metric_name, metric in quality_metrics:
                try:
                    metric.measure(test_case_obj)
                    print(f"{metric_name}: {metric.score:.2f}")
                except Exception as e:
                    print(f"{metric_name} failed: {str(e)}")
            
            print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ G-Eval test failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_g_eval_metrics() 