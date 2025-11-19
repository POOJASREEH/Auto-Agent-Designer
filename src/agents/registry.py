# src/agents/registry.py
"""
Registry mapping roles / names to concrete agent classes.
Used by simulator to instantiate correct class for each AgentSpec.
"""
from .simple_agent import SimpleAgent
from .planner_agent import PlannerAgent
from .classifier_agent import ClassifierAgent
from .executor_agent import ExecutorAgent

AGENT_REGISTRY = {
    "simple": SimpleAgent,
    "planner": PlannerAgent,
    "classifier": ClassifierAgent,
    "executor": ExecutorAgent,
}

