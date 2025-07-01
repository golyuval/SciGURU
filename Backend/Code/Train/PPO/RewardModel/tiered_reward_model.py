import numpy as np
from deepeval.test_case import LLMTestCase

# Import metrics from their respective modules
from Backend.Code.Eval.g_eval.zemla_metrics import (
    articulation_metric_explicit,
    internal_coherence_metric_explicit,
    completeness_metric_explicit,
    audience_awareness_metric_explicit,
    perceived_truth_metric_explicit
)
from Backend.Code.Eval.g_eval.explanation_quality import (
    analogy_metric,
    explanation_type_metric,
    connection_to_everyday_life_metric,
    correctness_metric,
    content_units_metric
)
from Backend.Code.Eval.Jargon.jargon_metric import JargonMetric

class TieredRewardModel:
    """
    A reward model that evaluates explanations based on a tiered system of importance:
    
    Tier 1 – Core Pedagogical Clarity & Structure (Most important):
    1) Articulation (Explicit) - Well-expressed explanations
    2) Internal Coherence (Explicit) - Parts fit together logically
    3) Completeness (Explicit) - No gaps in explanation
    4) Audience Awareness - Appropriate for general audience with basic scientific knowledge
    5) Higher jargon score (less jargon) - Accessible language
    
    Tier 2 – Cognitive Tools & Teaching Aids:
    6) Analogy - Use of analogies to enhance understanding
    7) Explanation Type - Preference for sophisticated explanations (Quasiscientific/transformative)
    8) Connection to Everyday Life - Relatable and grounded
    
    Tier 3 – Secondary Structure or Quantity Aids:
    9) Correctness - Factual accuracy
    10) Perceived Truth - Believability
    11) Content Units - Amount of information
    """
    
    def __init__(self):
        # Initialize all metrics
        self.metrics = {
            # Tier 1 - Core Pedagogical Clarity & Structure
            "articulation": articulation_metric_explicit,
            "internal_coherence": internal_coherence_metric_explicit,
            "completeness": completeness_metric_explicit,
            "audience_awareness": audience_awareness_metric_explicit,
            "jargon": JargonMetric(threshold=0.5),  # Higher score means less jargon
            
            # Tier 2 - Cognitive Tools & Teaching Aids
            "analogy": analogy_metric,
            "explanation_type": explanation_type_metric,
            "connection_to_everyday": connection_to_everyday_life_metric,
            
            # Tier 3 - Secondary Structure or Quantity Aids
            "correctness": correctness_metric,
            "perceived_truth": perceived_truth_metric_explicit,
            "content_units": content_units_metric
        }
        
        # Assign weights to each tier and metrics within tiers
        self.tier_weights = {
            "tier1": 0.6,  # 60% of total weight
            "tier2": 0.3,  # 30% of total weight
            "tier3": 0.1   # 10% of total weight
        }
        
        # Assign weights to individual metrics within tiers
        self.metric_weights = {
            # Tier 1 - Core Pedagogical Clarity & Structure
            "articulation": 0.25,        # 25% of tier 1
            "internal_coherence": 0.25,  # 25% of tier 1 
            "completeness": 0.20,        # 20% of tier 1
            "audience_awareness": 0.15,  # 15% of tier 1
            "jargon": 0.15,              # 15% of tier 1
            
            # Tier 2 - Cognitive Tools & Teaching Aids
            "analogy": 0.40,             # 40% of tier 2
            "explanation_type": 0.35,    # 35% of tier 2
            "connection_to_everyday": 0.25, # 25% of tier 2
            
            # Tier 3 - Secondary Structure or Quantity Aids
            "correctness": 0.40,         # 40% of tier 3
            "perceived_truth": 0.35,     # 35% of tier 3
            "content_units": 0.25        # 25% of tier 3
        }
    
    def calculate_reward(self, test_case: LLMTestCase) -> float:
        """
        Calculate the reward based on the tiered evaluation system.
        
        Args:
            test_case: An LLMTestCase containing the input, expected output, and actual output
            
        Returns:
            A float value representing the reward score (0-10)
        """
        scores = {}
        
        # Calculate scores for each metric
        for metric_name, metric in self.metrics.items():
            try:
                if metric_name == "jargon":
                    # For jargon, higher score is better (less jargon)
                    scores[metric_name] = metric.measure(test_case) * 10  # Scale to 0-10
                else:
                    # For G-Eval metrics
                    scores[metric_name] = metric.measure(test_case)
            except Exception as e:
                print(f"Error evaluating {metric_name}: {str(e)}")
                scores[metric_name] = 0.0
        
        # Calculate tier scores
        tier1_score = (
            scores["articulation"] * self.metric_weights["articulation"] +
            scores["internal_coherence"] * self.metric_weights["internal_coherence"] +
            scores["completeness"] * self.metric_weights["completeness"] +
            scores["audience_awareness"] * self.metric_weights["audience_awareness"] +
            scores["jargon"] * self.metric_weights["jargon"]
        )
        
        tier2_score = (
            scores["analogy"] * self.metric_weights["analogy"] +
            scores["explanation_type"] * self.metric_weights["explanation_type"] +
            scores["connection_to_everyday"] * self.metric_weights["connection_to_everyday"]
        )
        
        tier3_score = (
            scores["correctness"] * self.metric_weights["correctness"] +
            scores["perceived_truth"] * self.metric_weights["perceived_truth"] +
            scores["content_units"] * self.metric_weights["content_units"] 
        )
        
        # Calculate final reward (0-10 scale)
        final_reward = (
            tier1_score * self.tier_weights["tier1"] +
            tier2_score * self.tier_weights["tier2"] +
            tier3_score * self.tier_weights["tier3"]
        )
        
        return final_reward
    
    def get_detailed_scores(self, test_case: LLMTestCase) -> dict:
        """
        Returns a detailed breakdown of scores for each metric and tier.
        
        Args:
            test_case: An LLMTestCase containing the input, expected output, and actual output
            
        Returns:
            A dictionary containing detailed scores
        """
        scores = {}
        
        # Calculate scores for each metric
        for metric_name, metric in self.metrics.items():
            try:
                if metric_name == "jargon":
                    # For jargon, higher score is better (less jargon)
                    scores[metric_name] = metric.measure(test_case) * 10  # Scale to 0-10
                else:
                    # For G-Eval metrics
                    scores[metric_name] = metric.measure(test_case)
            except Exception as e:
                print(f"Error evaluating {metric_name}: {str(e)}")
                scores[metric_name] = 0.0
        
        # Calculate tier scores
        tier1_score = (
            scores["articulation"] * self.metric_weights["articulation"] +
            scores["internal_coherence"] * self.metric_weights["internal_coherence"] +
            scores["completeness"] * self.metric_weights["completeness"] +
            scores["audience_awareness"] * self.metric_weights["audience_awareness"] +
            scores["jargon"] * self.metric_weights["jargon"]
        )
        
        tier2_score = (
            scores["analogy"] * self.metric_weights["analogy"] +
            scores["explanation_type"] * self.metric_weights["explanation_type"] +
            scores["connection_to_everyday"] * self.metric_weights["connection_to_everyday"]
        )
        
        tier3_score = (
            scores["correctness"] * self.metric_weights["correctness"] +
            scores["perceived_truth"] * self.metric_weights["perceived_truth"] +
            scores["content_units"] * self.metric_weights["content_units"] 
        )
        
        # Calculate total tier weights
        tier1_weighted = tier1_score * self.tier_weights["tier1"]
        tier2_weighted = tier2_score * self.tier_weights["tier2"]
        tier3_weighted = tier3_score * self.tier_weights["tier3"]
        
        # Calculate final reward 
        final_reward = tier1_weighted + tier2_weighted + tier3_weighted
        
        return {
            "final_reward": final_reward,
            "tier_scores": {
                "tier1": {
                    "score": tier1_score,
                    "weighted_contribution": tier1_weighted,
                    "metrics": {
                        "articulation": scores["articulation"],
                        "internal_coherence": scores["internal_coherence"],
                        "completeness": scores["completeness"],
                        "audience_awareness": scores["audience_awareness"],
                        "jargon": scores["jargon"]
                    }
                },
                "tier2": {
                    "score": tier2_score,
                    "weighted_contribution": tier2_weighted,
                    "metrics": {
                        "analogy": scores["analogy"],
                        "explanation_type": scores["explanation_type"],
                        "connection_to_everyday": scores["connection_to_everyday"]
                    }
                },
                "tier3": {
                    "score": tier3_score,
                    "weighted_contribution": tier3_weighted,
                    "metrics": {
                        "correctness": scores["correctness"],
                        "perceived_truth": scores["perceived_truth"],
                        "content_units": scores["content_units"]
                    }
                }
            },
            "raw_scores": scores
        } 