# src/eval/simulator.py
from typing import List, Dict
from agents.base import AgentSpec
from agents.registry import AGENT_REGISTRY
from agents.simple_agent import SimpleAgent
import random

def _instantiate(spec: AgentSpec):
    role_key = spec.role.strip().lower()
    # pick class by role mapping
    if "planner" in role_key:
        cls = AGENT_REGISTRY.get("planner", SimpleAgent)
    elif "class" in role_key or "moderate" in role_key:
        cls = AGENT_REGISTRY.get("classifier", SimpleAgent)
    elif "exec" in role_key or "executor" in role_key:
        cls = AGENT_REGISTRY.get("executor", SimpleAgent)
    else:
        cls = AGENT_REGISTRY.get("simple", SimpleAgent)
    return cls(spec)

def run_simulator(specs: List[AgentSpec]) -> Dict:
    """
    Instantiate agents from specs, run each agent's test_cases and return a leaderboard.
    """
    random.seed(1234)
    results = []
    for s in specs:
        agent = _instantiate(s)
        tests_run = 0
        score = 0
        trace = []
        for tc in s.test_cases:
            tests_run += 1
            input_data = {"task": tc.get("task", ""), "input": tc.get("input", "")}
            try:
                out = agent.act(input_data)
                ok = bool(out.get("ok", False))
                if ok:
                    score += 1
            except Exception as e:
                out = {"response": None, "ok": False, "error": str(e)}
            trace.append({"task": tc.get("task"), "in": tc.get("input"), "out": out})
        results.append({
            "name": s.name,
            "role": s.role,
            "tests": tests_run,
            "score": score,
            "trace": trace
        })
    results_sorted = sorted(results, key=lambda r: r["score"], reverse=True)
    return {"leaderboard": results_sorted}
