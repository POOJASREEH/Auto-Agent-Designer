# src/eval/leaderboard.py
from typing import Dict, List
import json

def pretty_print(results: Dict):
    lb = results.get("leaderboard", [])
    for i, r in enumerate(lb, 1):
        print(f"{i}. {r['name']} (role={r['role']}) score={r['score']} / tests={r['tests']}")
    print("\nFull JSON:")
    print(json.dumps(results, indent=2))

