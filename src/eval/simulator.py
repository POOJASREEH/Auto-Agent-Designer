# src/eval/simulator.py
"""
Enhanced Simulator with Multi-Agent Pipeline Support (permanent version)

Backwards-compatible:
- Returns a top-level {"leaderboard": ...} like before
- Adds:
    {
      "individual_tests": { "per_agent": {...}, "leaderboard": [...] },
      "pipeline": { "final_output": ..., "trace": [...] },
      "leaderboard": [...]
    }

Important:
- CoordinatorAgent is NOT treated as a normal agent in individual tests
- Coordinator is constructed only for the pipeline path with (specs, objects)
"""

from typing import List, Dict, Any
import random

from agents.base import AgentSpec
from agents.registry import get_agent_class

# Coordinator is optional; import defensively
try:
    from agents.coordinator_agent import CoordinatorAgent  # type: ignore
    _HAVE_COORDINATOR = True
except Exception:  # pragma: no cover
    CoordinatorAgent = None
    _HAVE_COORDINATOR = False


# ----------------------------------------------------------------------------- #
# Instantiation helpers
# ----------------------------------------------------------------------------- #
def _instantiate(spec: AgentSpec):
    """
    Instantiate a normal agent (single-arg constructor).
    Skips CoordinatorAgent here; it's handled by the pipeline path.
    """
    cls = get_agent_class(spec)
    if CoordinatorAgent and cls is CoordinatorAgent:
        return None  # skip in individual tests
    return cls(spec)


def _instantiate_all(specs: List[AgentSpec]) -> Dict[str, Any]:
    """
    Build a dict {agent_name: agent_instance} for pipeline orchestration.
    Skips CoordinatorAgent; it's created separately with (specs, objects).
    """
    objs: Dict[str, Any] = {}
    for s in specs:
        cls = get_agent_class(s)
        if CoordinatorAgent and cls is CoordinatorAgent:
            continue
        try:
            objs[s.name] = cls(s)
        except Exception:
            # Keep simulator resilient even if one agent fails to construct
            continue
    return objs


# ----------------------------------------------------------------------------- #
# Individual test execution (keeps legacy scoring)
# ----------------------------------------------------------------------------- #
def _run_individual_tests(specs: List[AgentSpec]) -> Dict[str, Any]:
    """
    Run each agent's test_cases independently and compute legacy scores.
    CoordinatorAgent is skipped in this phase.
    """
    random.seed(1234)
    per_agent_results = []
    per_agent_map: Dict[str, Any] = {}

    for s in specs:
        agent = _instantiate(s)
        if agent is None:
            # likely CoordinatorAgent; skip scoring as a normal agent
            continue

        tests_run = 0
        score = 0
        trace = []

        for tc in getattr(s, "test_cases", []):
            tests_run += 1
            input_data = {"task": tc.get("task", ""), "input": tc.get("input", "")}
            try:
                out = agent.act(input_data)
                ok = bool(out.get("ok", False))
                if ok:
                    score += 1
            except Exception as e:
                out = {"response": None, "ok": False, "error": str(e)}
            trace.append({"task": tc.get("task"), "in": tc.get("input"), "out": out})

        rec = {
            "name": s.name,
            "role": s.role,
            "tests": tests_run,
            "score": score,
            "trace": trace,
        }
        per_agent_results.append(rec)
        per_agent_map[s.name] = trace

    leaderboard = sorted(per_agent_results, key=lambda r: r["score"], reverse=True)

    return {
        "per_agent": per_agent_map,   # {agent_name: [trace entries...]}
        "leaderboard": leaderboard,   # legacy-compatible
    }


# ----------------------------------------------------------------------------- #
# Multi-agent pipeline (Coordinator)
# ----------------------------------------------------------------------------- #
def _run_pipeline(specs: List[AgentSpec]) -> Dict[str, Any]:
    """
    Execute a sequential pipeline using CoordinatorAgent if available.
    Flow: Planner → Classifier → Extractor → Summarizer → Critic → Executor
    Only agents present will be included.
    """
    if not _HAVE_COORDINATOR:
        return {"final_output": None, "trace": [], "note": "CoordinatorAgent not available"}

    # Build object map (no coordinator here)
    obj_map = _instantiate_all(specs)

    # Coordinator takes (specs, obj_map) and runs the pipeline
    coord = CoordinatorAgent(specs, obj_map)
    out = coord.run_pipeline(initial_input="start")

    # out = {"final_output": <...>, "trace": [ {agent, task, input, output}, ... ]}
    return out


# ----------------------------------------------------------------------------- #
# Public API
# ----------------------------------------------------------------------------- #
def run_simulator(specs: List[AgentSpec]) -> Dict[str, Any]:
    """
    Unified simulator entrypoint.
    Returns:
        {
          "individual_tests": {
              "per_agent": {agent_name: trace_list, ...},
              "leaderboard": [ ...sorted by score desc... ]
          },
          "pipeline": {
              "final_output": ...,
              "trace": [ ... ],
              "note": "CoordinatorAgent not available"  # optional
          },
          "leaderboard": [ ... ]   # top-level legacy mirror
        }
    """
    indiv = _run_individual_tests(specs)
    pipe = _run_pipeline(specs)

    # Keep top-level 'leaderboard' for backward compatibility
    return {
        "individual_tests": indiv,
        "pipeline": pipe,
        "leaderboard": indiv["leaderboard"],
    }
