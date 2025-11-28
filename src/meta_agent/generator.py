# src/meta_agent/generator.py
"""
Meta-Agent Generator
--------------------

Supports two modes:
  ✔ Evaluator / Offline mode (default): uses DummyLLM
  ✔ Real LLM mode (optional): --provider {openai, gemini, groq}

Examples:
    # Evaluator-friendly (no keys required)
    python -m src.meta_agent.generator --mission data/missions.yml

    # With real LLM (optional)
    export OPENAI_API_KEY=sk-...
    python -m src.meta_agent.generator --mission data/missions.yml --provider openai
"""

from dataclasses import asdict
from typing import List
import argparse
import json
import yaml

from agents.base import AgentSpec
from meta_agent.llm_client import make_llm


class MetaAgentGenerator:
    def __init__(self, provider: str | None = None):
        """
        provider: "dummy", "openai", "gemini", "groq"
        None → default to DummyLLM (safe for evaluators)
        """
        self._client = make_llm(provider)

    # -------------------------
    # Load mission file
    # -------------------------
    def load_missions(self, path: str):
        with open(path, "r") as f:
            return yaml.safe_load(f)

    # -------------------------
    # Generate AgentSpec objects
    # -------------------------
    def generate_specs(self, mission_text: str) -> List[AgentSpec]:
        raw_specs = self._client.generate_agent_design(mission_text)
        specs: List[AgentSpec] = []

        for r in raw_specs:
            specs.append(
                AgentSpec(
                    name=r.get("name", "Agent"),
                    role=r.get("role", "Simple"),
                    tools=r.get("tools", []),
                    prompt=r.get("prompt", ""),
                    test_cases=r.get("test_cases", []),
                )
            )

        return specs

    # -------------------------
    # Full run from missions.yml
    # -------------------------
    def run_from_file(self, mission_path: str) -> List[AgentSpec]:
        missions = self.load_missions(mission_path)
        if not missions:
            raise ValueError("No missions found in the YAML file.")
        mission_text = missions[0].get("description", "")
        return self.generate_specs(mission_text)


# ---------------------------------------------------
# CLI ENTRY
# ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--mission",
        required=True,
        help="Path to missions.yml",
    )

    parser.add_argument(
        "--provider",
        choices=["dummy", "openai", "gemini", "groq"],
        default="dummy",
        help="LLM provider (default dummy; no keys required)",
    )

    args = parser.parse_args()

    # Build generator
    gen = MetaAgentGenerator(provider=args.provider)

    # Generate agent specs
    specs = gen.run_from_file(args.mission)

    # Print agent specs
    print("\n=== Agent Specs ===")
    print(json.dumps([asdict(s) for s in specs], indent=2))

    # Run simulation
    from eval.simulator import run_simulator
    results = run_simulator(specs)

    print("\n=== Simulation Results ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
