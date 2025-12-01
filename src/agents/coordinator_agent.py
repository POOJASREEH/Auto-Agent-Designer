# src/agents/coordinator_agent.py
"""
CoordinatorAgent — Orchestrates a multi-agent pipeline.

Fixes included:
- Proper normalization of outputs between steps using _to_scalar_text()
- Prevents downstream agents from receiving dicts/lists they cannot parse
- Ensures pipeline always flows smoothly
"""

from typing import List, Dict, Any
from agents.base import AgentSpec


class CoordinatorAgent:
    """
    Coordinator that runs a sequential pipeline:
    Planner → Classifier → Extractor → Summarizer → Critic → Executor
    Only agents that exist in the project will be called.

    Args:
        agent_specs  : list of AgentSpec
        agent_objects: {name: instantiated agent object}
    """

    ORDER = [
        "PlannerAgent",
        "ClassifierAgent",
        "ExtractorAgent",
        "SummarizerAgent",
        "CriticAgent",
        "ExecutorAgent",
    ]

    def __init__(self, agent_specs: List[AgentSpec], agent_objects: Dict[str, Any]):
        self.specs = agent_specs
        self.objects = agent_objects
        self.available_names = set(agent_objects.keys())

    # ----------------------------------------------------------------------
    # Normalize any agent's output into clean SCALAR TEXT for next step
    # ----------------------------------------------------------------------
    def _to_scalar_text(self, value):
        """
        Convert prior agent output → friendly string.

        Handles:
        - {"plan": [...]} → bullet list
        - list → joined lines
        - dict → compact JSON
        - numbers, booleans, strings → str()
        """
        try:
            # PLAN case: {"plan": [...]}
            if isinstance(value, dict) and "plan" in value and isinstance(value["plan"], list):
                lines = [f"- {str(x)}" for x in value["plan"]]
                return "Plan:\n" + "\n".join(lines)

            # List case
            if isinstance(value, list):
                return "\n".join(f"- {str(x)}" for x in value)

            # Generic dict → JSON
            if isinstance(value, dict):
                import json
                return json.dumps(value, ensure_ascii=False)

            # Scalar
            return str(value)

        except Exception:
            return str(value)

    # ----------------------------------------------------------------------
    # Main pipeline execution
    # ----------------------------------------------------------------------
    def run_pipeline(self, initial_input="start") -> Dict[str, Any]:
        trace = []
        current_input = initial_input

        for role_name in self.ORDER:

            if role_name not in self.available_names:
                # Skip missing agents
                continue

            agent = self.objects.get(role_name)
            if agent is None:
                continue

            # Build task input
            inp = {
                "task": "pipeline_step",
                "input": current_input
            }

            # Execute
            try:
                output = agent.act(inp)
            except Exception as e:
                output = {"response": None, "ok": False, "error": str(e)}

            trace.append({
                "agent": role_name,
                "task": "pipeline_step",
                "input": current_input,
                "output": output
            })

            # Feed clean, normalized output into next step
            if isinstance(output, dict):
                raw = output.get("response", output)
            else:
                raw = output

            current_input = self._to_scalar_text(raw)

        # Final output of pipeline
        return {
            "final_output": current_input,
            "trace": trace
        }
