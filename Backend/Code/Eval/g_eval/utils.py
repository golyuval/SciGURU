import json
from typing import Dict, Any

def save_evaluation_results(results: Dict[str, Any], filepath: str):
    """Save evaluation results to a JSON file"""
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)

def load_evaluation_results(filepath: str) -> Dict[str, Any]:
    """Load evaluation results from a JSON file"""
    with open(filepath, 'r') as f:
        return json.load(f)