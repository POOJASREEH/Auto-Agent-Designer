# tests/test_metrics.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from eval.metrics import compute_primary

def test_compute_primary_empty():
    res = {"leaderboard": []}
    m = compute_primary(res)
    assert m["mission_success_rate"] == 0.0

def test_compute_primary_values():
    res = {"leaderboard": [
        {"name":"A","score":1},
        {"name":"B","score":0},
        {"name":"C","score":2},
    ]}
    m = compute_primary(res)
    # two agents have score>0 out of 3 -> 2/3
    assert abs(m["mission_success_rate"] - (2/3)) < 1e-6
