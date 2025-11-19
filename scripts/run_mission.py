#!/usr/bin/env python3
"""
scripts/run_mission.py

Run an entire mission end-to-end:
- Load mission from missions.yml
- Generate AgentSpecs using the MetaAgentGenerator
- Simulate with run_simulator()
- Print leaderboard + metrics

Usage:
    python scripts/run_mission.py --mission data/missions.yml
"""

import sys, os, json, argparse
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from meta_agent.generator import MetaAgentGenerator
from eval.simulator import run_simulator
from eval.evaluator import Evaluator
from eval.leaderboard import pretty_print


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", required=True, help="Path to missions.yml")
    args = parser.parse_args()

    print("\n=== Auto-Agent Designer: Running Mission ===\n")

    generator = MetaAgentGenerator()
    specs = generator.run_from_file(args.mission)

    print("Generated Agents:")
    for s in specs:
        print(f" - {s.name} ({s.role})")

    print("\n=== Running Simulator ===")
    results = run_simulator(specs)

    print("\n=== Leaderboard ===")
    pretty_print(results)

    evaluator = Evaluator()
    metrics = evaluator.evaluate(results)

    print("\n=== Evaluation Metrics ===")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
