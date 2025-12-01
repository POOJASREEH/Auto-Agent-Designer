# src/agents/registry.py
"""
Registry mapping agent names/roles to concrete classes.
Used by simulator to instantiate the correct class for each AgentSpec.

Backwards-compatible:
- Still supports role-based fallback (simple/planner/classifier/executor)
- Adds name-based mapping (PlannerAgent, ClassifierAgent, etc.)
- Optional CoordinatorAgent (if file exists)
"""

from .simple_agent import SimpleAgent
from .planner_agent import PlannerAgent
from .classifier_agent import ClassifierAgent
from .executor_agent import ExecutorAgent

# Optional agents (present in your repo according to tree)
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

# Optional Coordinator (B5)
try:
    from .coordinator_agent import CoordinatorAgent  # type: ignore
except Exception:  # pragma: no cover
    CoordinatorAgent = None


# -----------------------------------------------------------------------------
# Name-based mapping (preferred when AgentSpec.name matches these keys)
# -----------------------------------------------------------------------------
AGENT_CLASSES = {
    "SimpleAgent": SimpleAgent,
    "PlannerAgent": PlannerAgent,
    "ClassifierAgent": ClassifierAgent,
    "ExecutorAgent": ExecutorAgent,
}

if SummarizerAgent:
    AGENT_CLASSES["SummarizerAgent"] = SummarizerAgent
if CriticAgent:
    AGENT_CLASSES["CriticAgent"] = CriticAgent
if ExtractorAgent:
    AGENT_CLASSES["ExtractorAgent"] = ExtractorAgent
if CoordinatorAgent:
    AGENT_CLASSES["CoordinatorAgent"] = CoordinatorAgent


# -----------------------------------------------------------------------------
# Role-keyword fallback (keeps old behavior working)
# -----------------------------------------------------------------------------
ROLE_FALLBACK = [
    ("planner", PlannerAgent),
    ("class", ClassifierAgent),      # matches 'classify', 'classifier'
    ("moderate", ClassifierAgent),
    ("exec", ExecutorAgent),         # matches 'exec', 'executor'
]

def get_agent_class_by_role(role_value: str):
    role_key = (role_value or "").strip().lower()
    for key, cls in ROLE_FALLBACK:
        if key in role_key:
            return cls
    return SimpleAgent


# -----------------------------------------------------------------------------
# Public helper
# -----------------------------------------------------------------------------
def get_agent_class(spec) -> type:
    """
    Resolve the best class for the given AgentSpec by:
      1) exact name match (PlannerAgent, ClassifierAgent, ...)
      2) role keyword fallback (planner/classifier/executor/...)
      3) SimpleAgent
    """
    # Prefer exact name match if it looks like 'PlannerAgent', etc.
    if getattr(spec, "name", None):
        cls = AGENT_CLASSES.get(spec.name)
        if cls:
            return cls

    # Otherwise fallback by role keywords
    return get_agent_class_by_role(getattr(spec, "role", "") )
