# src/agents/simple_agent.py
from typing import Dict
from .base import SpecialistAgent, AgentSpec

class SimpleAgent(SpecialistAgent):
    """
    Minimal deterministic agent used for Day1 demo.
    Behaviour is heuristic and deterministic for reproducibility.
    """
    def act(self, input_data: Dict[str, any]) -> Dict[str, any]:
        task = input_data.get("task", "")
        content = input_data.get("input") or input_data.get("text") or ""
        role = self.spec.role.lower()

        # Classifier heuristic
        if "classify" in role or "classifier" in role or "moderate" in role:
            label = "positive" if "good" in content.lower() or "ok" in content.lower() else "negative"
            return {"response": label, "ok": True, "meta": {"heuristic": "contains_good_word"}}

        # Planner heuristic
        if "plan" in role or "planner" in role:
            plan = [
                "inspect input",
                "enumerate candidate actions",
                "pick highest-priority action",
                "execute action with executor agent"
            ]
            return {"response": {"plan": plan}, "ok": True}

        # Executor heuristic
        if "exec" in role or "executor" in role:
            # Simulate action success
            return {"response": f"executed {task}", "ok": True}

        # Default echo
        return {"response": f"{self.spec.name} handled task {task}", "ok": True}
