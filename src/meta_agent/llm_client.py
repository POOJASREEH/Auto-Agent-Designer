# src/meta_agent/llm_client.py
"""
Improved LLM Client Layer with Telemetry + Mission-Aware Test Case Engine (B4)

Features:
- Intelligent DummyLLMClient (offline, deterministic, enhanced test cases)
- Mission-aware test-case generation (positive, negative, edge, stress)
- Optional real LLM support: OpenAI, Gemini, Groq
- Telemetry tracking:
      last_prompt
      last_response
      last_latency_ms
      last_tokens_prompt / completion
      last_cost_usd
- make_llm(provider) factory for generator.py
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
    """Extract JSON array from messy LLM output."""
    text = text.strip()
    try:
        data = json.loads(text)
        if isinstance(data, list):
            return data
    except Exception:
        pass

    # fallback: extract bracketed list
    m = re.search(r"\[.*\]", text, flags=re.S)
    if m:
        try:
            data = json.loads(m.group(0))
            if isinstance(data, list):
                return data
        except Exception:
            pass

    return []


def _mk_test(task: str, inp):
    return {"task": task, "input": inp}


# ============================================================
# Base Interface + Telemetry
# ============================================================

class BaseLLMClient:
    """Parent class for Dummy and real LLM clients."""

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
        return {
            "prompt_tokens": self.last_tokens_prompt,
            "completion_tokens": self.last_tokens_completion,
            "latency_ms": self.last_latency_ms,
            "cost_usd": self.last_cost_usd,
        }


# ============================================================
# DUMMY LLM (Enhanced - Mission Aware Test Cases)
# ============================================================

class DummyLLMClient(BaseLLMClient):
    """
    Enhanced offline meta-agent designer.
    - Rule-based, deterministic
    - Mission-aware test-case generation (B4)
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

    # ============================================================
    # B4: Mission-Aware Test Case Engine
    # ============================================================

    def _build_test_cases(self, agent_name: str, role: str, mission: str):
        """
        Generates:
        - positive cases
        - negative cases
        - edge cases
        - stress tests (use mission content)
        """
        if role == "Planner":
            positives = [
                _mk_test("create_plan", "Plan a 3-step workflow"),
                _mk_test("refine_plan", "Improve clarity in step 2"),
            ]
            negatives = [
                _mk_test("create_plan", ""),
                _mk_test("refine_plan", "12345"),
            ]
            edge = [_mk_test("create_plan", "Plan only one step")]
            stress = [_mk_test("create_plan", "Plan details: " + mission[:200])]

        elif role == "Classifier":
            positives = [
                _mk_test("classify", "I absolutely love this!"),
                _mk_test("classify", "You are stupid."),
                _mk_test("classify", "This is okay, not great."),
            ]
            negatives = [_mk_test("classify", ""), _mk_test("classify", 12345)]
            edge = [_mk_test("classify", "....???!!!")]
            stress = [_mk_test("classify", mission[:250])]

        elif role == "Summarizer":
            positives = [
                _mk_test("summarize", "Model accuracy improved 70→78%."),
                _mk_test("summarize", "Revenue increased 12% this quarter."),
            ]
            negatives = [_mk_test("summarize", "")]
            edge = [_mk_test("summarize", "A")]
            stress = [_mk_test("summarize", mission * 3)]

        elif role == "Critic":
            positives = [_mk_test("evaluate", "Plan: collect → analyze → deploy")]
            negatives = [_mk_test("evaluate", "")]
            edge = [_mk_test("evaluate", "StepStepStep")]
            stress = [_mk_test("evaluate", mission[:250])]

        elif role == "Extractor":
            positives = [
                _mk_test("extract", "Alice met Bob at Google in 2024"),
                _mk_test("extract", "Python created by Guido in 1991"),
            ]
            negatives = [_mk_test("extract", "")]
            edge = [_mk_test("extract", "....")]
            stress = [_mk_test("extract", mission[:250])]

        elif role == "Executor":
            positives = [_mk_test("execute", "step1"), _mk_test("execute", "step2")]
            negatives = [_mk_test("execute", "unknown"), _mk_test("execute", "")]
            edge = [_mk_test("execute", "STEP1")]
            stress = [_mk_test("execute", mission.split(" ")[0])]

        else:
            positives = [_mk_test("task", "example input")]
            negatives = [_mk_test("task", "")]
            edge = [_mk_test("task", "...")]
            stress = [_mk_test("task", mission)]

        tests = positives + negatives + edge + stress

        # ensure uniqueness
        seen, unique = set(), []
        for t in tests:
            key = (t["task"], str(t["input"]))
            if key not in seen:
                seen.add(key)
                unique.append(t)

        return unique[:10]

    # ============================================================
    # Agent Templates (Now Using _build_test_cases)
    # ============================================================

    def _planner(self, mission):
        return {
            "name": "PlannerAgent",
            "role": "Planner",
            "tools": ["notebook", "scheduler"],
            "prompt": f"You are a planner. Break mission into steps.\nMission:\n{mission}",
            "test_cases": self._build_test_cases("PlannerAgent", "Planner", mission),
        }

    def _classifier(self, mission):
        return {
            "name": "ClassifierAgent",
            "role": "Classifier",
            "tools": ["simple_model"],
            "prompt": "Classify text: positive/negative/toxic/neutral.",
            "test_cases": self._build_test_cases("ClassifierAgent", "Classifier", mission),
        }

    def _summarizer(self, mission):
        return {
            "name": "SummarizerAgent",
            "role": "Summarizer",
            "tools": ["notebook"],
            "prompt": "Summarize into 3 bullets: key point, risk, next step.",
            "test_cases": self._build_test_cases("SummarizerAgent", "Summarizer", mission),
        }

    def _critic(self, mission):
        return {
            "name": "CriticAgent",
            "role": "Critic",
            "tools": ["rulebook"],
            "prompt": "Evaluate answers. Give score (0–5) + 2 improvements.",
            "test_cases": self._build_test_cases("CriticAgent", "Critic", mission),
        }

    def _extractor(self, mission):
        return {
            "name": "ExtractorAgent",
            "role": "Extractor",
            "tools": ["regex"],
            "prompt": "Extract keywords from text.",
            "test_cases": self._build_test_cases("ExtractorAgent", "Extractor", mission),
        }

    def _executor(self, mission):
        return {
            "name": "ExecutorAgent",
            "role": "Executor",
            "tools": ["shell", "api"],
            "prompt": "Execute simple commands. Return executed:<task>.",
            "test_cases": self._build_test_cases("ExecutorAgent", "Executor", mission),
        }

    # ============================================================
    # Main Logic
    # ============================================================

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

        # dedupe
        final, seen = [], set()
        for a in agents:
            if a["name"] not in seen:
                seen.add(a["name"])
                final.append(a)

        if len(final) < 2:
            final.append(self._executor(mission_text))

        # telemetry
        self.last_response = f"{len(final)} dummy agents generated"
        self.last_latency_ms = int((time.time() - start) * 1000)
        self.last_tokens_prompt = len(mission_text.split())

        return final


# ============================================================
# REAL LLM CLIENTS (OpenAI / Gemini / Groq)
# ============================================================

class OpenAIClient(BaseLLMClient):
    def __init__(self, model: str = None):
        super().__init__()
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY not set")
        try:
            from openai import OpenAI
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
            usage = rsp.usage or {}
            self.last_tokens_prompt = usage.get("prompt_tokens", 0)
            self.last_tokens_completion = usage.get("completion_tokens", 0)
            self.last_response = text or ""

        except Exception:
            dummy = DummyLLMClient()
            return dummy.generate_agent_design(mission_text)

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

        except Exception:
            dummy = DummyLLMClient()
            return dummy.generate_agent_design(mission_text)

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
            self.last_tokens_prompt = usage.get("prompt_tokens", 0)
            self.last_tokens_completion = usage.get("completion_tokens", 0)
            self.last_response = text

        except Exception:
            dummy = DummyLLMClient()
            return dummy.generate_agent_design(mission_text)

        self.last_latency_ms = int((time.time() - start) * 1000)
        data = _safe_json_from_text(text or "")
        return data or DummyLLMClient().generate_agent_design(mission_text)


# ============================================================
# LLM FACTORY
# ============================================================

def make_llm(provider: str | None) -> BaseLLMClient:
    p = (provider or "dummy").lower()

    if p == "openai":
        return OpenAIClient()
    if p == "gemini":
        return GeminiClient()
    if p == "groq":
        return GroqClient()

    return DummyLLMClient()
