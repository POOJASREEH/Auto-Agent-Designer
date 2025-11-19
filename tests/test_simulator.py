from eval.simulator import run_simulator
from agents.base import AgentSpec

def test_simulator_basic():
    specs = [AgentSpec(name="A", role="r", tools=[], prompt="p", test_cases=[{"task":"t"}])]
    out = run_simulator(specs)
    assert "leaderboard" in out
    assert out["leaderboard"][0]["score"] == 1
