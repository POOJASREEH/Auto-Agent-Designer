# tests/test_base_agent.py
import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "src")))

import pytest
from agents.base import AgentSpec, SpecialistAgent

def test_agentspec_fields():
    spec = AgentSpec(name="A", role="R", tools=["t"], prompt="p", test_cases=[{"task":"t"}])
    assert spec.name == "A"
    assert spec.role == "R"
    assert isinstance(spec.test_cases, list)

def test_specialistagent_abstract():
    spec = AgentSpec(name="A", role="R", tools=[], prompt="", test_cases=[])
    agent = SpecialistAgent(spec)
    with pytest.raises(NotImplementedError):
        agent.act({"task":"x"})
