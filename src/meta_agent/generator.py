# src/meta_agent/generator.py
import yaml
import argparse
import json
from dataclasses import asdict
from typing import List
import random
from agents.base import AgentSpec

SEED = 1234
random.seed(SEED)

class DummyLLM:
    """
    Deterministic, rule-based 'LLM' for Day1 so you don't need external keys.
    Generates a small team of agents for a mission text.
    """
    def generate_agent_specs(self, mission_text: str) -> List[AgentSpec]:
        specs = []
        mission_lower = mission_text.lower()
        # Always create a planner
        specs.append(AgentSpec(
            name="PlannerAgent",
            role="Planner",
            tools=["notebook", "scheduler"],
            prompt=f"Plan steps for: {mission_text}",
            test_cases=[{"task": "create_plan", "input":"short"}]
        ))
        # Create a classifier if mission mentions classify/moderation
        if any(w in mission_lower for w in ["classify","moderate","toxic","comments","detect"]):
            specs.append(AgentSpec(
                name="ClassifierAgent",
                role="Classifier",
                tools=["simple_model"],
                prompt=f"Classify inputs for: {mission_text}",
                test_cases=[{"task":"classify","input":"sample comment: this is good"}]
            ))
        # Create an executor (general)
        specs.append(AgentSpec(
            name="ExecutorAgent",
            role="Executor",
            tools=["shell","api"],
            prompt=f"Execute actions for: {mission_text}",
            test_cases=[{"task":"execute","input":"step1"}]
        ))
        # Always return at least 2 agents
        return specs

def load_missions(path: str):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def run(mission_path: str, use_llm: bool=False):
    missions = load_missions(mission_path)
    mission = missions[0]
    mission_text = mission.get("description", "")
    if use_llm:
        # Placeholder: we still use DummyLLM unless you plug a real client
        llm = DummyLLM()
    else:
        llm = DummyLLM()
    specs = llm.generate_agent_specs(mission_text)
    # Print specs as JSON
    out = [asdict(s) for s in specs]
    print("=== Generated Agent Specs ===")
    print(json.dumps(out, indent=2))
    # Run simulator on generated specs
    from eval.simulator import run_simulator
    results = run_simulator(specs)
    print("=== Simulation Results ===")
    print(json.dumps(results, indent=2))
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", required=True, help="path to missions.yml")
    parser.add_argument("--use-llm", action="store_true", help="use LLM client (placeholder)")
    args = parser.parse_args()
    run(args.mission, use_llm=args.use_llm)
