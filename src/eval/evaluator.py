# src/eval/evaluator.py
from typing import Dict, List
from .metrics import compute_primary

class Evaluator:
    def __init__(self):
        pass

    def evaluate(self, results: Dict) -> Dict:
        """
        Compute evaluation metrics and return a summary dict.
        """
        metrics = compute_primary(results)
        metrics["num_agents"] = len(results.get("leaderboard", []))
        metrics["top_agent"] = results.get("leaderboard", [None])[0]
        return metrics

