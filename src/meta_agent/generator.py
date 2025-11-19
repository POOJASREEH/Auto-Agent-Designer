# src/meta_agent/generator.py
"""
Meta-agent generator: given a mission spec, produces AgentSpec objects.
By default uses DummyLLM for deterministic behaviour. Plug real LLM via llm_client.py later.
"""
from typing import List, Dict, Any
from dataclasses import asdict
import yaml
import json
import os
import random

from agents.base import AgentSpec

SEED = 1234
random.seed(SEED)

class DummyLLM:
    """
    Simple deterministic rule-based generator to avoid requiring API keys.
    """
    def generate(self, mission_text: str) -> List[Dict[str, Any]]:
        mission_lower = mission_text.lower()
        specs = []

        # Planner always useful
        specs.append({
            "name": "PlannerAgent",
            "role": "Planner",
            "tools": ["notebook", "scheduler"],
            "prompt": f"Plan steps for: {mission_text}",
            "test_cases": [{"task": "create_plan", "input": mission_text}]
        })

        # Classifier if mission involves moderation/text classification
        if any(k in mission_lower for k in ["classify", "moderate", "toxic", "comments", "detect"]):
            specs.append({
                "name": "ClassifierAgent",
                "role": "Classifier",
                "tools": ["simple_model"],
                "prompt": f"Classify inputs for: {mission_text}",
                "test_cases": [{"task": "classify", "input": "this is good"}]
            })

        # Executor always useful
        specs.append({
            "name": "ExecutorAgent",
            "role": "Executor",
            "tools": ["shell", "api"],
            "prompt": f"Execute steps for: {mission_text}",
            "test_cases": [{"task": "execute", "input": "step1"}]
        })

        # Ensure at least two agents returned
        if len(specs) < 2:
            specs.append({
                "name": "HelperAgent",
                "role": "Simple",
                "tools": [],
                "prompt": f"Assist with: {mission_text}",
                "test_cases": [{"task": "assist", "input": ""}]
            })

        return specs

class MetaAgentGenerator:
    def __init__(self, llm_client=None):
        # llm_client may be provided later; default to DummyLLM
        self.llm = llm_client or DummyLLM()

    def load_missions(self, path: str):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    def generate_specs(self, mission_text: str) -> List[AgentSpec]:
        raw = self.llm.generate(mission_text)
        specs = []
        for r in raw:
            specs.append(AgentSpec(
                name=r["name"],
                role=r.get("role", "Simple"),
                tools=r.get("tools", []),
                prompt=r.get("prompt", ""),
                test_cases=r.get("test_cases", [])
            ))
        return specs

    def run_from_file(self, mission_path: str):
        missions = self.load_missions(mission_path)
        if not missions:
            raise ValueError("No missions found in file")
        mission = missions[0]
        text = mission.get("description", "")
        specs = self.generate_specs(text)
        return specs

# CLI helper if run as module
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", required=True, help="Path to missions.yml")
    args = parser.parse_args()
    mg = MetaAgentGenerator()
    specs = mg.run_from_file(args.mission)
    print("=== Agent Specs ===")
    print(json.dumps([asdict(s) for s in specs], indent=2))
    # run simulator
    from eval.simulator import run_simulator
    results = run_simulator(specs)
    print("=== Simulation Results ===")
    print(json.dumps(results, indent=2))

if __name__ == "__main__":
    main()
