**Auto-Agent-Designer**

A Meta-Agent System That Designs, Evaluates, and Coordinates Specialist AI Agents

Overview

Auto-Agent-Designer is a meta-agent framework that automatically generates a team of specialist agents—planners, classifiers, extractors, critics, executors, and more—based on a high-level mission description.
It features an offline DummyLLM, optional OpenAI/Gemini/Groq integration, a multi-agent Coordinator, and a fully instrumented evaluation + simulation pipeline.

This system is built for experimentation, research, rapid prototyping, and competition environments (e.g., Kaggle Auto-Agents). It prioritizes repeatability, transparency, and zero-dependency offline operation while allowing seamless upgrades to real LLM providers.

Key Features
Meta-Agent Generator

Converts mission text into a set of structured AgentSpec objects.

Uses DummyLLM by default (offline, deterministic).

Supports OpenAI, Gemini, and Groq for real LLM generation.

Specialist Agents

Includes:

PlannerAgent

ClassifierAgent

SummarizerAgent

CriticAgent

ExtractorAgent

ExecutorAgent

CoordinatorAgent (orchestrates the full agent pipeline)

Each agent has tools, prompts, and multiple test cases.

Coordinator Pipeline

The CoordinatorAgent connects all agents into a sequential reasoning workflow:

Planner → Classifier → Extractor → Critic → Executor


The pipeline produces:

Final output

Full step-by-step trace

Agent-level success/failure signals

Simulator + Leaderboard

Runs each agent on its own test cases.

Executes the full Coordinator pipeline.

Produces:

Individual test performance

Pipeline trace

Metrics

Leaderboard (sorted by score)

Reproducible Artifacts

Exports a JSON bundle containing:

Generated agent specs

Individual test results

Coordinator pipeline trace

Leaderboard

Telemetry (LLM provider, tokens, latency, etc.)
