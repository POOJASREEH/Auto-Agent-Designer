# src/eval/simulator.py
"""
Enhanced Simulator with Multi-Agent Pipeline Support (B5)

Backwards-compatible:
- Still returns {"leaderboard": ...} for older notebook/tests
- Adds:
    {
      "individual_tests": { ... per-agent results ... },
      "pipeline": { "final_output": ..., "trace": [...] },
      "leaderboard": [ ... sorted by score ... ]
    }
"""

from typing import List, Dict, Any
import random

from agents.base import AgentSpec
from agents.registry import get_agent_class, AGENT_CLASSES

# Coordinator is optional; import defensively
try:
    from agents.coordinator_agent import CoordinatorAgent  # type: ignore
    _HAVE_COORDINATOR = True
except Exception:  # pragma: no cover
    CoordinatorAgent = None
    _HAVE_COORDINATOR = False


# -----------------------------------------------------------------------------
# Instantiation
# -----------------------------------------------------------------------------
def _instantiate(spec: AgentSpec):
    """
    Instantiate an agent class for the provided AgentSpec.
    Uses name-based mapping first, then role-based fallback (via registry.get_agent_class).
    """
    cls = get_agent_class(spec)
    return cls(spec)


def _instantiate_all(specs: List[AgentSpec]) -> Dict[str, Any]:
    """
    Build a dict {agent_name: agent_instance} for convenience.
    """
    objs = {}
    for s in specs:
        try:
            agent = _instantiate(s)
            objs[s.name] = agent
        except Exception as e:
            # Skip problematic specs but keep the simulator running
            # (You can also add logging here.)
            continue
    return objs


# -----------------------------------------------------------------------------
# Individual test execution (keeps old scoring behavior)
# -----------------------------------------------------------------------------
def _run_individual_tests(specs: List[AgentSpec]) -> Dict[str, Any]:
    """
    Run each agent's test_cases independently, return a dict of per-agent results
    and the legacy 'leaderboard' list sorted by score (desc).
    """
    random.seed(1234)
    per_agent_results = []
    per_agent_map: Dict[str, Any] = {}

    for s in specs:
        agent = _instantiate(s)
        tests_run = 0
        score = 0
        trace = []

        for tc in s.test_cases:
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


# -----------------------------------------------------------------------------
# Multi-agent pipeline (Coordinator)
# -----------------------------------------------------------------------------
def _run_pipeline(specs: List[AgentSpec]) -> Dict[str, Any]:
    """
    Execute a sequential pipeline using CoordinatorAgent if available.
    Flow: Planner → Classifier → Extractor → Summarizer → Critic → Executor
    Only agents present will be included.
    """
    if not _HAVE_COORDINATOR:
        return {"final_output": None, "trace": [], "note": "CoordinatorAgent not available"}

    # build objects keyed by name
    obj_map = _instantiate_all(specs)

    # Coordinator takes specs + object map and runs a pipeline
    coord = CoordinatorAgent(specs, obj_map)
    out = coord.run_pipeline(initial_input="start")

    # out: {"final_output": <...>, "trace": [ {agent, task, input, output}, ... ]}
    return out


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
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
