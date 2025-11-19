# tests/test_simulator.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agents.base import AgentSpec
from eval.simulator import run_simulator

def test_simulator_basic():
    specs = [
        AgentSpec(name="A", role="Classifier", tools=[], prompt="", test_cases=[{"task":"classify","input":"good"}]),
        AgentSpec(name="B", role="Executor", tools=[], prompt="", test_cases=[{"task":"execute","input":"step1"},{"task":"execute","input":"step2"}]),
    ]
    res = run_simulator(specs)
    assert "leaderboard" in res
    lb = res["leaderboard"]
    assert isinstance(lb, list)
    # A should score 1, B should score 2 (both tests pass)
    scores = {r["name"]: r["score"] for r in lb}
    assert scores["A"] == 1
    assert scores["B"] == 2

def test_simulator_trace_contains_out():
    specs = [AgentSpec(name="A", role="Simple", tools=[], prompt="", test_cases=[{"task":"t","input":"x"}])]
    res = run_simulator(specs)
    trace = res["leaderboard"][0]["trace"]
    assert isinstance(trace, list)
    assert "out" in trace[0]
