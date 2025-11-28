# Auto-Agent Designer 

A meta-agent that automatically designs, configures, and evaluates** specialist AI agents from a mission description.  
Lightweight, reproducible, and Kaggle-friendly.

## What it does
1. Reads a mission (e.g., “moderate toxic comments”).
2. Generates a small team of agents (Planner / Classifier / Executor) as `AgentSpec`s.
3. Simulates each agent against simple test cases.
4. Ranks agents on a leaderboard and outputs metrics.

## 🔧 Quickstart
```bash
# 1) Install deps
pip install -r requirements.txt

# 2) Run the generator + simulator (uses DummyLLM; no API keys)
python -m src.meta_agent.generator --mission data/missions.yml

# 3) Run all tests
pytest -q
