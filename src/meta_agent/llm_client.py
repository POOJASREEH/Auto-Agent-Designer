# src/meta_agent/llm_client.py
"""
Simple pluggable LLM client wrapper.
By default we keep this minimal. You may add OpenAI or Google calls here later.
"""
import os
from typing import List

class BaseLLMClient:
    def generate_agent_design(self, mission_text: str) -> List[dict]:
        raise NotImplementedError

class DummyLLMClient(BaseLLMClient):
    def generate_agent_design(self, mission_text: str) -> List[dict]:
        # alias for compatibility with generator.DummyLLM
        from .generator import DummyLLM
        return DummyLLM().generate(mission_text)

# Example OpenAI client stub (fill API key & uncomment to use)
class OpenAIClient(BaseLLMClient):
    def __init__(self, openai_api_key: str):
        self.api_key = openai_api_key
        try:
            import openai
            openai.api_key = openai_api_key
            self._openai = openai
        except Exception:
            self._openai = None

    def generate_agent_design(self, mission_text: str) -> List[dict]:
        if not self._openai:
            raise RuntimeError("openai package not available")
        prompt = f"Design agents for mission: {mission_text}"
        resp = self._openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            max_tokens=512
        )
        # The response parsing logic depends on the prompt you design.
        # For now, return empty list (user will implement advanced parsing).
        return []

