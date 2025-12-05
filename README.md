<img width="2912" height="1440" alt="Gemini_Generated_Image_2t2gq92t2gq92t2g-min" src="https://github.com/user-attachments/assets/64bfea08-b710-408d-a3c2-5e9ec13fd884" />

# **Auto-Agent-Designer**  
*A Meta-Agent System That Designs, Evaluates, and Coordinates Specialist AI Agents*

---

## **Overview**
Auto-Agent-Designer is a meta-agent framework that automatically **generates, simulates, and evaluates a team of AI specialist agents** from a single mission description. It is designed for, research experimentation, workflow automation, and rapid prototyping.

The system includes a fully offline **DummyLLM**, a powerful **CoordinatorAgent**, and a complete evaluation suite with metrics, traces, and leaderboard generation.  
It is built for **repeatability, interpretability, and modularity**, making it easy to extend, modify, or integrate into other agent ecosystems.

---

## **Key Features**

### **Meta-Agent Generator**
- Converts mission text → structured multi-agent team (`AgentSpec` objects).
- Offline DummyLLM for deterministic agent design.
- Optional real LLM support (OpenAI, Gemini, Groq).
- Generates: roles, tools, prompts, test cases.

### **Specialist Agent Set**
Includes:
- **PlannerAgent**
- **ClassifierAgent**
- **SummarizerAgent**
- **CriticAgent**
- **ExtractorAgent**
- **ExecutorAgent**
- **CoordinatorAgent** (orchestrates full pipeline)

Each agent is modular, testable, and built on top of a common `SpecialistAgent` base class.

### **Coordinator Pipeline**
The CoordinatorAgent manages the execution flow by linking all specialist agents into a seamless and logically connected chain. Instead of agents working in isolation, the coordinator ensures that the output of one agent becomes the input for the next, creating a complete end-to-end workflow. This structured pipeline improves interpretability, consistency, and mission success rates. At the end of the pipeline, a final consolidated output is produced along with a detailed step-by-step execution trace.

### **Simulator & Evaluation Engine**
The built-in simulation engine runs all agents through two layers of evaluation:
1. Individual Agent Tests – Each agent executes its own curated test cases to confirm its role-specific behavior.
2. Full Pipeline Simulation – All agents are executed sequentially under the CoordinatorAgent to determine how well they collaborate.
This produces a robust evaluation report including:
- **Success/failure counts**
- **Per-agent performance summaries**
- **Pipeline trace**
- **Structured metrics**
- **A ranked leaderboard**

### **Wrapping Up**
  Auto-Agent-Designer provides a complete, flexible, and reproducible environment for building intelligent multi-agent workflows from scratch. By combining offline determinism (via DummyLLM), modular specialist agents, a structured Coordinator pipeline, and a rigorous evaluation engine, the system enables researchers, developers, and students to design and test complex agent behavior with ease. Its compatibility with both offline and real LLM providers ensures that the framework remains accessible while still supporting advanced experimentation. Whether used for academic research, workflow automation, prototyping intelligent systems, Auto-Agent-Designer delivers a powerful balance of reliability, transparency, and extensibility. The project is actively maintained and fully open-source, inviting developers to explore, contribute, and build upon the meta-agent paradigm.
