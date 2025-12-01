# src/agents/coordinator_agent.py
"""
CoordinatorAgent — Multi-Agent Orchestration (B5)

Purpose
-------
Run a simple, deterministic pipeline across the agents produced by the meta-agent.
The coordinator:
  • Chooses a sensible step order
  • For each agent, picks a default task (from its first test_case or 'run')
  • Calls agent.act(...) using the same dict shape as the simulator
  • Feeds each agent's output into the next as input
  • Records a detailed trace for later visualization

This file is evaluator-safe (no external deps) and works offline.
"""

from typing import Any, Dict, List


class CoordinatorAgent:
    def __init__(self, agent_specs: List[Any], agent_objects: Dict[str, Any]):
        """
        Parameters
        ----------
        agent_specs : list of AgentSpec
            Dataclass instances describing each agent (name, role, tools, prompt, test_cases).
        agent_objects : dict[str, Agent]
            Instantiated agent objects, keyed by agent name. Each object exposes .act(input_dict).
        """
        self.agent_specs = agent_specs
        self.agent_objects = agent_objects
        # quick index for spec lookup by name
        self._spec_by_name = {getattr(s, "name", ""): s for s in agent_specs}
        self.trace: List[Dict[str, Any]] = []

    def _default_task_for(self, agent_name: str) -> str:
        """Pick a default task for the agent (first test case if available, else 'run')."""
        spec = self._spec_by_name.get(agent_name)
        if spec and getattr(spec, "test_cases", None):
            tc0 = spec.test_cases[0]
            return str(tc0.get("task", "run"))
        return "run"

    def _append_trace(self, agent_name: str, task: str, input_value: Any, output: Dict[str, Any]):
        self.trace.append(
            {
                "agent": agent_name,
                "task": task,
                "input": input_value,
                "output": output,
            }
        )

    def run_pipeline(self, initial_input: Any = "start") -> Dict[str, Any]:
        """
        Execute a simple, fixed-order pipeline using whichever agents exist:

            Planner → Classifier → Extractor → Summarizer → Critic → Executor

        Only agents present in agent_objects are executed.
        The output of one step becomes the input to the next step.

        Returns
        -------
        dict with:
            final_output : str|dict|Any   (best-effort scalarization)
            trace        : list of {agent, task, input, output}
        """
        order = [
            "PlannerAgent",
            "ClassifierAgent",
            "ExtractorAgent",
            "SummarizerAgent",
            "CriticAgent",
            "ExecutorAgent",
        ]

        current_input: Any = initial_input

        for name in order:
            agent = self.agent_objects.get(name)
            if agent is None:
                continue  # skip missing agents gracefully

            task = self._default_task_for(name)

            # Act uses the SAME shape as your simulator: {"task": ..., "input": ...}
            try:
                output = agent.act({"task": task, "input": current_input})
            except Exception as e:
                output = {"response": None, "ok": False, "error": str(e)}

            # Log the step
            self._append_trace(name, task, current_input, output)

            # Feed response to next step
            # Prefer a clean scalar for "current_input" if available.
            if isinstance(output, dict):
                # Use 'response' if present; otherwise pass the whole dict forward
                next_input = output.get("response", output)
            else:
                next_input = output
            current_input = next_input

        return {"final_output": current_input, "trace": self.trace}
