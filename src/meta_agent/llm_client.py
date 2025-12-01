# src/meta_agent/llm_client.py
"""
Improved LLM Client Layer with Telemetry

Features:
- Intelligent DummyLLMClient (offline, deterministic)
- Optional real LLM support: OpenAI, Gemini, Groq
- Telemetry tracking (eval/logging safe):
      - last_prompt
      - last_response
      - last_latency_ms
      - last_tokens_prompt / last_tokens_completion
      - last_cost_usd (LLM only)
- make_llm(provider) factory used by generator.py

Providers:
  - dummy   (default for evaluator)
  - openai
  - gemini
  - groq
"""

from __future__ import annotations
import json
import os
import re
import time
from typing import List, Dict


# ============================================================
# Utility helpers
# ============================================================

def _safe_json_from_text(text: str) -> List[dict]:
    """Extract a JSON array from messy LLM output. Always returns a list."""
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # fallback: extract `[ ... ]`
    m = re.search(r"\[.*\]", text, flags=re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except Exception:
            pass

    return []


def _mk_test(task: str, inp: str):
    return {"task": task, "input": inp}


# ============================================================
# Base Interface + Telemetry
# ============================================================

class BaseLLMClient:
    """
    All LLM clients (dummy or real) must set the following telemetry fields:

    self.last_prompt
    self.last_response
    self.last_latency_ms
    self.last_tokens_prompt
    self.last_tokens_completion
    self.last_cost_usd
    """

    def __init__(self):
        self.last_prompt = ""
        self.last_response = ""
        self.last_latency_ms = 0
        self.last_tokens_prompt = 0
        self.last_tokens_completion = 0
        self.last_cost_usd = 0.0

    def _reset_stats(self):
        self.last_prompt = ""
        self.last_response = ""
        self.last_latency_ms = 0
        self.last_tokens_prompt = 0
        self.last_tokens_completion = 0
        self.last_cost_usd = 0.0

    def generate_agent_design(self, mission_text: str) -> List[dict]:
        raise NotImplementedError

    def last_usage(self):
        """Return usage stats as a dictionary."""
        return {
            "prompt_tokens": self.last_tokens_prompt,
            "completion_tokens": self.last_tokens_completion,
            "latency_ms": self.last_latency_ms,
            "cost_usd": self.last_cost_usd,
        }


# ============================================================
# IMPROVED DUMMY LLM (Evaluator-safe)
# ============================================================

class DummyLLMClient(BaseLLMClient):
    """
    Stronger offline meta-agent designer.

    - Deterministic patterns
    - Rule-based multi-agent design
    - More realistic prompts + test cases
    - Zero dependencies, zero cost
    """

    def __init__(self):
        super().__init__()

    def _detect_intents(self, text: str) -> Dict[str, bool]:
        t = text.lower()
        return {
            "moderate": any(k in t for k in ["toxic", "abuse", "flag", "moderate"]),
            "classify": any(k in t for k in ["classify", "label", "detect"]),
            "summarize": any(k in t for k in ["summarize", "digest", "overview"]),
            "plan": any(k in t for k in ["plan", "workflow", "roadmap"]),
            "critic": any(k in t for k in ["score", "evaluate", "improve"]),
            "execute": any(k in t for k in ["execute", "perform", "apply"]),
            "extract": any(k in t for k in ["extract", "keyword", "entity"]),
        }

    # ---- Agent templates ----

    def _planner(self, mission):
        return {
            "name": "PlannerAgent",
            "role": "Planner",
            "tools": ["notebook", "scheduler"],
            "prompt": f"You are a planner. Break mission into steps.\nMission:\n{mission}",
            "test_cases": [
                _mk_test("create_plan", "draft 3-step plan"),
                _mk_test("refine_plan", "improve clarity"),
            ],
        }

    def _classifier(self, mission):
        return {
            "name": "ClassifierAgent",
            "role": "Classifier",
            "tools": ["simple_model"],
            "prompt": "Classify text: positive/negative/toxic/neutral.",
            "test_cases": [
                _mk_test("classify", "i love this"),
                _mk_test("classify", "you idiot"),
                _mk_test("classify", "average quality"),
            ],
        }

    def _summarizer(self, mission):
        return {
            "name": "SummarizerAgent",
            "role": "Summarizer",
            "tools": ["notebook"],
            "prompt": "Summarize into 3 bullets: key point, risk, next step.",
            "test_cases": [_mk_test("summarize", "accuracy improved 70→78")],
        }

    def _critic(self, mission):
        return {
            "name": "CriticAgent",
            "role": "Critic",
            "tools": ["rulebook"],
            "prompt": "Evaluate answers. Give score (0–5) + 2 improvements.",
            "test_cases": [_mk_test("evaluate", "Plan: collect → train → deploy")],
        }

    def _extractor(self, mission):
        return {
            "name": "ExtractorAgent",
            "role": "Extractor",
            "tools": ["regex"],
            "prompt": "Extract keywords from text.",
            "test_cases": [_mk_test("extract", "Alice met Bob at Google 2024")],
        }

    def _executor(self, mission):
        return {
            "name": "ExecutorAgent",
            "role": "Executor",
            "tools": ["shell", "api"],
            "prompt": "Execute simple commands. Return executed:<task>.",
            "test_cases": [
                _mk_test("execute", "step1"),
                _mk_test("execute", "invalid"),
            ],
        }

    # ---- Main Logic ----

    def generate_agent_design(self, mission_text: str) -> List[dict]:
        start = time.time()

        self._reset_stats()
        self.last_prompt = mission_text

        intents = self._detect_intents(mission_text)
        agents = [self._planner(mission_text)]

        if intents["classify"] or intents["moderate"]:
            agents.append(self._classifier(mission_text))
        if intents["summarize"]:
            agents.append(self._summarizer(mission_text))
        if intents["critic"]:
            agents.append(self._critic(mission_text))
        if intents["extract"]:
            agents.append(self._extractor(mission_text))
        if intents["plan"] or intents["execute"]:
            agents.append(self._executor(mission_text))

        # ensure uniqueness
        final, seen = [], set()
        for a in agents:
            if a["name"] not in seen:
                seen.add(a["name"])
                final.append(a)

        # at least 2 agents
        if len(final) < 2:
            final.append(self._executor(mission_text))

        # telemetry
        self.last_response = f"{len(final)} dummy agents generated"
        self.last_latency_ms = int((time.time() - start) * 1000)
        self.last_tokens_prompt = len(mission_text.split())
        self.last_tokens_completion = 0
        self.last_cost_usd = 0.0

        return final


# ============================================================
# OPTIONAL REAL LLM CLIENTS (OpenAI, Gemini, Groq)
# ============================================================

class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = None):
        super().__init__()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")

        try:
            from openai import OpenAI  # type: ignore
        except ImportError:
            raise RuntimeError("pip install openai")

        self.client = OpenAI(api_key=api_key)
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    def generate_agent_design(self, mission_text: str) -> List[dict]:
        self._reset_stats()
        self.last_prompt = mission_text
        start = time.time()

        prompt = f"""
Return STRICT JSON array of agent specs:
name, role, tools[], prompt, test_cases[] ({{task,input}})
Mission:
{mission_text}
"""

        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(os.getenv("TEMPERATURE", "0.2")),
            )

            text = rsp.choices[0].message.content

            # telemetry
            usage = rsp.usage or {}
            self.last_tokens_prompt = usage.get("prompt_tokens", 0)
            self.last_tokens_completion = usage.get("completion_tokens", 0)
            self.last_cost_usd = 0.0  # Kaggle safe (no official pricing)
            self.last_response = text or ""

        except Exception:
            # fallback to Dummy
            dummy = DummyLLMClient()
            out = dummy.generate_agent_design(mission_text)
            self.last_response = "fallback_dummy"
            self.last_latency_ms = dummy.last_latency_ms
            return out

        self.last_latency_ms = int((time.time() - start) * 1000)

        data = _safe_json_from_text(text or "")
        return data or DummyLLMClient().generate_agent_design(mission_text)


class GeminiClient(BaseLLMClient):
    def __init__(self, model: str = None):
        super().__init__()
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise RuntimeError("GOOGLE_API_KEY not set")

        try:
            import google.generativeai as genai
        except ImportError:
            raise RuntimeError("pip install google-generativeai")

        genai.configure(api_key=api_key)
        self.genai = genai
        self.model = model or os.getenv("GEMINI_MODEL", "gemini-1.5-flash")

    def generate_agent_design(self, mission_text: str) -> List[dict]:
        self._reset_stats()
        self.last_prompt = mission_text
        start = time.time()

        prompt = f"Return STRICT JSON array of agent specs.\nMission:\n{mission_text}"

        try:
            model = self.genai.GenerativeModel(self.model)
            rsp = model.generate_content(prompt)
            text = rsp.text or ""

            self.last_response = text
            self.last_tokens_prompt = 0  # Gemini Python SDK doesn't expose tokens
            self.last_tokens_completion = 0

        except Exception:
            dummy = DummyLLMClient()
            out = dummy.generate_agent_design(mission_text)
            self.last_response = "fallback_dummy"
            self.last_latency_ms = dummy.last_latency_ms
            return out

        self.last_latency_ms = int((time.time() - start) * 1000)
        data = _safe_json_from_text(text)
        return data or DummyLLMClient().generate_agent_design(mission_text)


class GroqClient(BaseLLMClient):
    def __init__(self, model: str = None):
        super().__init__()
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise RuntimeError("GROQ_API_KEY not set")

        try:
            from groq import Groq
        except ImportError:
            raise RuntimeError("pip install groq")

        self.client = Groq(api_key=api_key)
        self.model = model or os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

    def generate_agent_design(self, mission_text: str) -> List[dict]:
        self._reset_stats()
        self.last_prompt = mission_text
        start = time.time()

        prompt = f"Return STRICT JSON array of agent specs.\nMission:\n{mission_text}"

        try:
            rsp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=float(os.getenv("TEMPERATURE", "0.2")),
            )

            text = rsp.choices[0].message.content
            usage = rsp.usage or {}

            # telemetry
            self.last_tokens_prompt = usage.get("prompt_tokens", 0)
            self.last_tokens_completion = usage.get("completion_tokens", 0)
            self.last_response = text

        except Exception:
            dummy = DummyLLMClient()
            out = dummy.generate_agent_design(mission_text)
            self.last_response = "fallback_dummy"
            self.last_latency_ms = dummy.last_latency_ms
            return out

        self.last_latency_ms = int((time.time() - start) * 1000)
        data = _safe_json_from_text(text or "")
        return data or DummyLLMClient().generate_agent_design(mission_text)


# ============================================================
# LLM FACTORY
# ============================================================

def make_llm(provider: str | None) -> BaseLLMClient:
    """
    provider = "dummy", "openai", "gemini", "groq"
    Evaluator always uses dummy.
    """
    p = (provider or "dummy").lower()

    if p == "openai":
        return OpenAIClient()
    if p == "gemini":
        return GeminiClient()
    if p == "groq":
        return GroqClient()

    return DummyLLMClient()
