# src/agents/__init__.py
"""
Public exports for the agents package.

- New API:
    AGENT_CLASSES (name → class)
    get_agent_class(spec)  # resolves best class from spec.name/role

- Backwards compatibility:
    AGENT_REGISTRY with keys: simple / planner / classifier / executor
      (so older code importing AGENT_REGISTRY keeps working)
"""

from .base import AgentSpec, SpecialistAgent
from .simple_agent import SimpleAgent
from .planner_agent import PlannerAgent
from .classifier_agent import ClassifierAgent
from .executor_agent import ExecutorAgent

# Optional agents (exist only if files present)
try:
    from .summarizer_agent import SummarizerAgent  # type: ignore
except Exception:  # pragma: no cover
    SummarizerAgent = None
try:
    from .critic_agent import CriticAgent  # type: ignore
except Exception:  # pragma: no cover
    CriticAgent = None
try:
    from .extractor_agent import ExtractorAgent  # type: ignore
except Exception:  # pragma: no cover
    ExtractorAgent = None

# New registry interface from registry.py
from .registry import AGENT_CLASSES, get_agent_class, get_agent_class_by_role  # type: ignore

# ----------------------------------------------------------------------
# Backwards-compatible minimal role map (used by older simulator/tests)
# ----------------------------------------------------------------------
AGENT_REGISTRY = {
    "simple": SimpleAgent,
    "planner": PlannerAgent,
    "classifier": ClassifierAgent,
    "executor": ExecutorAgent,
}

# Export optional classes into AGENT_CLASSES (if available)
if SummarizerAgent:
    AGENT_CLASSES["SummarizerAgent"] = SummarizerAgent
if CriticAgent:
    AGENT_CLASSES["CriticAgent"] = CriticAgent
if ExtractorAgent:
    AGENT_CLASSES["ExtractorAgent"] = ExtractorAgent

__all__ = [
    "AgentSpec",
    "SpecialistAgent",
    "SimpleAgent",
    "PlannerAgent",
    "ClassifierAgent",
    "ExecutorAgent",
    "AGENT_CLASSES",
    "get_agent_class",
    "get_agent_class_by_role",
    "AGENT_REGISTRY",  # legacy alias
]
