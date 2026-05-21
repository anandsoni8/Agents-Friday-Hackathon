# TotalAI: Multi-Agent Agentic RAG Assistant 🚀

TotalAI is an advanced Retrieval-Augmented Generation (RAG) application that moves beyond standard "blind retrieval" by orchestrating multiple AI Agents. Built for AI hackathons, it features strict context bounding, hallucination prevention, and real-time LLM observability.

## Features
*   **Multi-Agent Workflow (LangGraph):** Utilizes a state-graph architecture to route tasks between specialized agents.
*   **Context Grader Agent:** Evaluates retrieved chunks *before* generation to prevent hallucinations.
*   **LLM-as-a-Judge Evaluator:** Scores final outputs in real-time on Faithfulness, Relevance, Accuracy, and Safety.
*   **Interactive UI:** Built with Streamlit for a seamless chat experience and live metrics dashboard.
*   **Local Vector Store:** Powered by ChromaDB for fast semantic search.

## Tech Stack
*   **Frontend:** Streamlit
*   **Orchestration:** LangChain, LangGraph
*   **Vector Database:** ChromaDB
*   **Embeddings & LLM:** OpenAI (Compatible with DeepSeek/Custom endpoints)

## Installation & Setup

1. **Clone the repository and navigate to the directory.**
2. **Activate your virtual environment:**
   
```bash
   myenv\Scripts\activate