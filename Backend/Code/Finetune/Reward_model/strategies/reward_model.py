import os
import sys
from typing import Dict, List, Optional
import pandas as pd
from deepeval.test_case import LLMTestCase
from dotenv import load_dotenv

# Add the parent directory to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))))

from Backend.Code.Finetune.Reward_model.strategies.eval.Jargon.jargon_util import analyze_text
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

class ScientificExplanationRewardModel:
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize the reward model with optional weights for different metrics.
        
        Args:
            weights: Dictionary of metric names and their weights in the final score.
                    If None, uses default weights.
        """
        # Default weights for different metrics
        self.weights = weights or {
            # Jargon metrics
            'jargon_score': 0.2,  # Lower is better (less jargon)
            
            # Zemla metrics
            'internal_coherence': 0.1,
            'completeness': 0.1,
            'alternatives': 0.05,
            'articulation': 0.1,
            'perceived_truth': 0.1,
            
            # Explanation quality metrics
            'explanation_type': 0.1,
            'correctness': 0.1,
            'metaphor': 0.05,
            'content_units': 0.05,
            'connection_to_everyday_life': 0.05,
            'analogy': 0.05
        }
        
        # Load required data for jargon analysis
        self._load_jargon_data()
        
    def _load_jargon_data(self):
        """Load required data for jargon analysis."""
        try:
            # Load names and words data
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            names_path = os.path.join(base_path, "eval", "Jargon", "names.csv")
            words_path = os.path.join(base_path, "eval", "Jargon", "DataUKUS2018-2021.csv")
            
            self.names = pd.read_csv(names_path, header=None)[0].tolist()
            self.words = pd.read_csv(words_path, header=None).set_index(0)[1].to_dict()
        except Exception as e:
            print(f"Error loading jargon data: {str(e)}")
            raise
    
    def _calculate_jargon_score(self, text: str) -> float:
        """Calculate the jargon score for a given text."""
        try:
            # Analyze text for jargon
            score = analyze_text(text, self.words, self.names)
            # Convert to reward (higher score = less jargon = better)
            return 1 - score
        except Exception as e:
            print(f"Error calculating jargon score: {str(e)}")
            return 0.0
    
    def _calculate_g_eval_scores(self, question: str, answer: str, expected_answer: Optional[str] = None) -> Dict[str, float]:
        """Calculate G-Eval scores for a given question-answer pair."""
        scores = {}
        
        # Create test case
        test_case = LLMTestCase(
            input=question,
            actual_output=answer,
            expected_output=expected_answer
        )
        
        # Zemla metrics
        metrics = [
            ('internal_coherence', internal_coherence_metric),
            ('completeness', completeness_metric),
            ('alternatives', alternatives_metric),
            ('articulation', articulation_metric),
            ('perceived_truth', perceived_truth_metric)
        ]
        
        for name, metric in metrics:
            try:
                metric.measure(test_case)
                scores[name] = metric.score / 10.0  # Normalize to 0-1
            except Exception as e:
                print(f"Error calculating {name}: {str(e)}")
                scores[name] = 0.0
        
        # Explanation quality metrics
        quality_metrics = [
            ('explanation_type', explanation_type_metric),
            ('correctness', correctness_metric),
            ('metaphor', metaphor_metric),
            ('content_units', content_units_metric),
            ('connection_to_everyday_life', connection_to_everyday_life_metric),
            ('analogy', analogy_metric)
        ]
        
        for name, metric in quality_metrics:
            try:
                metric.measure(test_case)
                scores[name] = metric.score / 10.0  # Normalize to 0-1
            except Exception as e:
                print(f"Error calculating {name}: {str(e)}")
                scores[name] = 0.0
        
        return scores
    
    def calculate_reward(self, 
                        question: str, 
                        answer: str, 
                        expected_answer: Optional[str] = None) -> float:
        """
        Calculate the final reward score for a given question-answer pair.
        
        Args:
            question: The input question
            answer: The model's answer
            expected_answer: Optional expected answer for correctness evaluation
            
        Returns:
            float: The final reward score (higher is better)
        """
        try:
            # Calculate jargon score
            jargon_score = self._calculate_jargon_score(answer)
            
            # Calculate G-Eval scores
            g_eval_scores = self._calculate_g_eval_scores(question, answer, expected_answer)
            
            # Combine scores using weights
            final_score = self.weights['jargon_score'] * jargon_score
            for metric, score in g_eval_scores.items():
                final_score += self.weights[metric] * score
            
            return final_score
            
        except Exception as e:
            print(f"Error calculating reward: {str(e)}")
            return 0.0
    
    def evaluate_batch(self, 
                      questions: List[str], 
                      answers: List[str], 
                      expected_answers: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Evaluate multiple question-answer pairs in batch.
        
        Args:
            questions: List of questions
            answers: List of answers
            expected_answers: Optional list of expected answers
            
        Returns:
            pd.DataFrame: DataFrame containing scores for each metric and the final reward
        """
        results = []
        
        for i, (question, answer) in enumerate(zip(questions, answers)):
            expected_answer = expected_answers[i] if expected_answers else None
            
            # Calculate scores
            jargon_score = self._calculate_jargon_score(answer)
            g_eval_scores = self._calculate_g_eval_scores(question, answer, expected_answer)
            
            # Calculate final reward
            final_score = self.weights['jargon_score'] * jargon_score
            for metric, score in g_eval_scores.items():
                final_score += self.weights[metric] * score
            
            # Combine all scores
            result = {
                'question': question,
                'answer': answer,
                'jargon_score': jargon_score,
                **g_eval_scores,
                'final_reward': final_score
            }
            
            if expected_answer:
                result['expected_answer'] = expected_answer
                
            results.append(result)
        
        return pd.DataFrame(results) 