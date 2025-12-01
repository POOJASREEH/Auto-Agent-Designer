# src/agents/coordinator_agent.py
"""
CoordinatorAgent — Orchestrates a multi-agent pipeline (permanent version)

- Normalizes outputs between steps so downstream agents receive readable text
- Uses first test_case's task (if present) as the step's task; otherwise a sensible default
- Deterministic, evaluator-safe, no external dependencies

Pipeline order (only runs agents that exist):
    Planner → Classifier → Extractor → Summarizer → Critic → Executor
"""

from typing import Any, Dict, List
from agents.base import AgentSpec


class CoordinatorAgent:
    ORDER = [
        "PlannerAgent",
        "ClassifierAgent",
        "ExtractorAgent",
        "SummarizerAgent",
        "CriticAgent",
        "ExecutorAgent",
    ]

    def __init__(self, agent_specs: List[AgentSpec], agent_objects: Dict[str, Any]):
        # Canonical attribute names used by simulator
        self.specs: List[AgentSpec] = agent_specs
        self.objects: Dict[str, Any] = agent_objects

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #
    def _default_task_for(self, agent_name: str) -> str:
        """Pick a default task for an agent (first test_case.task if available)."""
        spec_map = {getattr(s, "name", ""): s for s in self.specs}
        spec = spec_map.get(agent_name)
        if spec and getattr(spec, "test_cases", None):
            return str(spec.test_cases[0].get("task", "pipeline_step"))
        return "pipeline_step"

    def _to_scalar_text(self, value: Any) -> str:
        """
        Convert an agent's output into a clean string for the next step.
        Handles:
            - {"plan": [...]}  →  "Plan:\n- item\n- item"
            - list             →  "- item\n- item"
            - dict             →  compact JSON
            - scalars          →  str(...)
        """
        try:
            if isinstance(value, dict) and "plan" in value and isinstance(value["plan"], list):
                return "Plan:\n" + "\n".join(f"- {str(x)}" for x in value["plan"])
            if isinstance(value, list):
                return "\n".join(f"- {str(x)}" for x in value)
            if isinstance(value, dict):
                import json
                return json.dumps(value, ensure_ascii=False)
            return str(value)
        except Exception:
            return str(value)

    # ------------------------------------------------------------------ #
    # Main
    # ------------------------------------------------------------------ #
    def run_pipeline(self, initial_input: Any = "start") -> Dict[str, Any]:
        """
        Execute the sequential pipeline across available agents.
        Returns:
            {
              "final_output": <str>,
              "trace": [ {agent, task, input, output}, ... ]
            }
        """
        trace: List[Dict[str, Any]] = []
        current_input: Any = initial_input

        for name in self.ORDER:
            agent = self.objects.get(name)
            if agent is None:
                continue

            task = self._default_task_for(name)

            # Simulator-compatible input shape
            step_input = {"task": task, "input": current_input}
            try:
                output = agent.act(step_input)
            except Exception as e:
                output = {"response": None, "ok": False, "error": str(e)}

            trace.append({
                "agent": name,
                "task": task,
                "input": current_input,
                "output": output,
            })

            # Normalize to clean text for next step
            raw = output.get("response", output) if isinstance(output, dict) else output
            current_input = self._to_scalar_text(raw)

        return {"final_output": current_input, "trace": trace}
