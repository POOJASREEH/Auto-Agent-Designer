#!/usr/bin/env python3
"""
scripts/export_results.py

Run a mission and export results as JSON into a results directory.

Usage:
    python scripts/export_results.py --mission data/missions.yml --out_dir results
"""

import sys, os, json, argparse, time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from meta_agent.generator import MetaAgentGenerator
from eval.simulator import run_simulator


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mission", required=True, help="Path to missions.yml")
    parser.add_argument("--out_dir", default="results", help="Where to save JSON results")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # run pipeline
    mg = MetaAgentGenerator()
    specs = mg.run_from_file(args.mission)
    results = run_simulator(specs)

    # timestamp filename
    timestamp = int(time.time())
    out_path = os.path.join(args.out_dir, f"results_{timestamp}.json")

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {out_path}")


if __name__ == "__main__":
    main()
