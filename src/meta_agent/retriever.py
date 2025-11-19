# src/meta_agent/retriever.py
"""
Simple file-based retriever to support RAG context later.
Searches for keyword matches in files under data/.
"""
import os
from typing import List

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

def retrieve(mission_text: str, top_k: int = 3) -> List[str]:
    mission_lower = mission_text.lower()
    hits = []
    if not os.path.isdir(DATA_DIR):
        return hits
    for root, _, files in os.walk(DATA_DIR):
        for fname in files:
            if not fname.endswith((".txt", ".md", ".yml", ".yaml")):
                continue
            path = os.path.join(root, fname)
            try:
                with open(path, "r", encoding="utf-8") as f:
                    txt = f.read().lower()
                if any(tok in txt for tok in mission_lower.split()[:5]):
                    hits.append(path)
            except Exception:
                continue
    return hits[:top_k]

