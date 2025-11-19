# tests/test_simple_agent.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

from agents.base import AgentSpec
from agents.simple_agent import SimpleAgent

def test_simple_agent_classifier():
    spec = AgentSpec(name="C", role="Classifier", tools=[], prompt="", test_cases=[])
    a = SimpleAgent(spec)
    out = a.act({"task":"classify", "input":"this is good"})
    assert out["ok"] is True
    assert out["response"] == "positive"

    out2 = a.act({"task":"classify", "input":"this is BAD"})
    assert out2["ok"] is True
    assert out2["response"] == "negative"

def test_simple_agent_planner_and_executor():
    spec_p = AgentSpec(name="P", role="Planner", tools=[], prompt="", test_cases=[])
    a_p = SimpleAgent(spec_p)
    out_p = a_p.act({"task":"create_plan", "input":"make a plan"})
    assert out_p["ok"] is True
    assert isinstance(out_p["response"], dict) or "plan" in str(out_p["response"]) or "inspect" in str(out_p["response"])

    spec_e = AgentSpec(name="E", role="Executor", tools=[], prompt="", test_cases=[])
    a_e = SimpleAgent(spec_e)
    out_e = a_e.act({"task":"execute", "input":"step1"})
    assert out_e["ok"] is True
    assert "executed" in out_e["response"]
