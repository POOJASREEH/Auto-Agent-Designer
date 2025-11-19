#!/usr/bin/env python3
"""
scripts/build_leaderboard.py

Build a global leaderboard by reading multiple JSON result files
from a "results" directory.

Usage:
    python scripts/build_leaderboard.py --results_dir results/
"""

import sys, os, json, argparse
from glob import glob

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", default="results", help="Directory containing results_*.json")
    args = parser.parse_args()

    pattern = os.path.join(args.results_dir, "results_*.json")
    files = glob(pattern)

    if not files:
        print(f"No result files found in {args.results_dir}.")
        return

    entries = []
    for f in files:
        with open(f, "r") as infile:
            data = json.load(infile)
            lb = data.get("leaderboard", [])
            entries.extend(lb)

    # sort all agents globally
    entries_sorted = sorted(entries, key=lambda r: r["score"], reverse=True)

    print("\n=== Global Leaderboard Across Missions ===\n")
    for i, e in enumerate(entries_sorted, 1):
        print(f"{i:02d}. {e['name']} ({e['role']})  score={e['score']} tests={e['tests']}")

    # save
    out_path = os.path.join(args.results_dir, "leaderboard_global.json")
    with open(out_path, "w") as out:
        json.dump(entries_sorted, out, indent=2)

    print(f"\nSaved global leaderboard to {out_path}")


if __name__ == "__main__":
    main()
