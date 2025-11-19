# src/eval/simulator.py
from typing import List, Dict
from agents.base import AgentSpec
from agents.simple_agent import SimpleAgent
import random

def run_simulator(specs: List[AgentSpec]) -> Dict:
    """
    Simple simulator: instantiate SimpleAgent for each spec,
    run its test_cases, and score as count of passed tests.
    """
    random.seed(1234)
    results = []
    for s in specs:
        agent = SimpleAgent(s)
        tests_run = 0
        score = 0
        trace = []
        for tc in s.test_cases:
            tests_run += 1
            task = tc.get("task", "")
            input_data = {"task": task, "input": tc.get("input")}
            out = agent.act(input_data)
            ok = out.get("ok", False)
            # For Day1: treat any ok True as pass
            if ok:
                score += 1
            trace.append({"task": task, "input": tc.get("input"), "out": out})
        results.append({
            "name": s.name,
            "role": s.role,
            "tests": tests_run,
            "score": score,
            "trace": trace
        })
    # rank by score desc
    results_sorted = sorted(results, key=lambda r: r["score"], reverse=True)
    return {"leaderboard": results_sorted}
