# src/agents/__init__.py
from .base import AgentSpec, SpecialistAgent
from .simple_agent import SimpleAgent
from .registry import AGENT_REGISTRY

__all__ = ["AgentSpec", "SpecialistAgent", "SimpleAgent", "AGENT_REGISTRY"]

