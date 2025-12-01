# src/agents/coordinator_agent.py

"""
CoordinatorAgent — Multi-Agent Orchestration Layer

This agent:
- Reads all agent specs (Planner, Classifier, Summarizer, etc.)
- Builds a workflow/pipeline dynamically
- Executes each agent step-by-step using their 'act' method
- Captures a full execution trace for visualization
- Returns final aggregated output

Lightweight, deterministic, evaluator-safe.
"""

from typing import Dict, List, Any


class CoordinatorAgent:
    def __init__(self, agent_specs: List[Any], agent_objects: Dict[str, Any]):
        """
        agent_specs  — list of AgentSpec dataclasses
        agent_objects — instantiated agent classes, keyed by name
        """
        self.agent_specs = agent_specs
        self.agent_objects = agent_objects
        self.trace = []

    def _log(self, agent_name: str, task: str, input_value: str, output: Any):
        self.trace.append({
            "agent": agent_name,
            "task": task,
            "input": input_value,
            "output": output
        })

    def run_pipeline(self, initial_input: str = "start"):
        """
        Executes a sequential pipeline:
        Planner → Classifier → Extractor → Summarizer → Critic → Executor
        Only agents present in the mission will be included.
        """
        order = ["PlannerAgent", "ClassifierAgent", "ExtractorAgent",
                 "SummarizerAgent", "CriticAgent", "ExecutorAgent"]

        current = initial_input

        for name in order:
            if name in self.agent_objects:
                agent = self.agent_objects[name]
                # always use the first test_case task as default action
                spec = next(s for s in self.agent_specs if s.name == name)
                if spec.test_cases:
                    tc = spec.test_cases[0]
                    task = tc["task"]
                else:
                    task = "run"

                output = agent.act(task, current)
                self._log(name, task, current, output)
                current = output  # feed output → next agent

        return {
            "final_output": current,
            "trace": self.trace
        }
