# src/agents/simple_agent.py
from .base import SpecialistAgent, AgentSpec

class SimpleAgent(SpecialistAgent):
    """
    Simple specialist agent for Day1:
    - If role contains 'classify' or 'classifier', do naive labeling.
    - Else echo the task.
    """
    def act(self, input_data):
        task = input_data.get("task", "")
        # Naive classification heuristic
        if "classify" in self.spec.role.lower() or "classifier" in self.spec.role.lower():
            # example heuristic: contains word 'good' => positive
            label = "positive" if "good" in task.lower() else "negative"
            return {"response": label, "ok": True}
        # Execution heuristic
        if "execute" in self.spec.role.lower() or "executor" in self.spec.role.lower():
            return {"response": f"executed:{task}", "ok": True}
        # Planner or other: return a tiny plan
        if "plan" in self.spec.role.lower() or "planner" in self.spec.role.lower():
            return {"response": f"plan: step1 -> inspect; step2 -> act on '{task}'", "ok": True}
        # default echo
        return {"response": f"{self.spec.name} done:{task}", "ok": True}

