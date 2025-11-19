# src/agents/base.py
from dataclasses import dataclass, field
from typing import Any, List, Dict

@dataclass
class AgentSpec:
    name: str
    role: str
    tools: List[str]
    prompt: str
    test_cases: List[Dict[str, Any]] = field(default_factory=list)

class SpecialistAgent:
    """
    Base class for specialist agents. MVP: deterministic/simple behavior.
    """
    def __init__(self, spec: AgentSpec):
        self.spec = spec

    def act(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use spec.prompt + input_data to return a dict with keys: response, ok
        For Day1, this is a simple deterministic echo / heuristic.
        """
        task = input_data.get("task", "")
        # Default behavior: echo with agent name
        return {"response": f"{self.spec.name} processed task '{task}'", "ok": True}

