from typing import List, Dict, Optional
import pandas as pd
from deepeval.test_case import LLMTestCase
from tqdm import tqdm

from .metrics.explanation_quality import explanation_type_metric
from .metrics.content_analysis import content_units_metric, connection_to_everyday_life_metric
from .metrics.correctness import correctness_metric

class GEvaluator:
    def __init__(self, metrics: Optional[List] = None):
        """Initialize the G-Evaluator with specified metrics"""
        self.metrics = metrics or [
            explanation_type_metric,
            content_units_metric,
            connection_to_everyday_life_metric,
            correctness_metric
        ]
        
    def evaluate_single(self, 
                       question: str, 
                       answer: str, 
                       expected_answer: Optional[str] = None) -> Dict:
        """Evaluate a single explanation"""
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_answer
        )
        
        results = {}
        for metric in self.metrics:
            try:
                metric.measure(test_case)
                results[metric.name] = {
                    'score': metric.score,
                    'reason': getattr(metric, 'reason', None)
                }
            except Exception as e:
                print(f"Error evaluating {metric.name}: {str(e)}")
                results[metric.name] = {
                    'score': None,
                    'reason': f"Error: {str(e)}"
                }
        
        return results
    
    def evaluate_batch(self, 
                      questions: List[str], 
                      answers: List[str], 
                      expected_answers: Optional[List[str]] = None) -> pd.DataFrame:
        """Evaluate multiple explanations in batch"""
        results = []
        
        for i, (question, answer) in enumerate(tqdm(zip(questions, answers))):
            expected_answer = expected_answers[i] if expected_answers else None
            evaluation = self.evaluate_single(question, answer, expected_answer)
            evaluation['question'] = question
            evaluation['answer'] = answer
            if expected_answer:
                evaluation['expected_answer'] = expected_answer
            results.append(evaluation)
        
        return pd.DataFrame(results)