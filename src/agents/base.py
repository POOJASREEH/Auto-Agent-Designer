# src/agents/base.py
from dataclasses import dataclass, field
from typing import Any, Dict, List

@dataclass
class AgentSpec:
    """
    Simple serializable spec for a specialist agent.
    """
    name: str
    role: str
    tools: List[str]
    prompt: str
    test_cases: List[Dict[str, Any]] = field(default_factory=list)

class SpecialistAgent:
    """
    Base class for all specialist agents. Subclasses implement act().
    """
    def __init__(self, spec: AgentSpec):
        self.spec = spec

    def act(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute one step for the given input_data.
        Must return dict with at least: {"response": ..., "ok": bool}
        """
        raise NotImplementedError("SpecialistAgent.act must be implemented by subclasses")
