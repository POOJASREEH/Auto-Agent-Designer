# src/meta_agent/memory.py
"""
Very small memory module used to keep session-level info.
"""
from typing import Dict, Any

class InMemoryMemory:
    def __init__(self):
        self._store = {}

    def write(self, key: str, value: Any):
        self._store[key] = value

    def read(self, key: str, default=None):
        return self._store.get(key, default)

    def dump(self):
        return dict(self._store)

