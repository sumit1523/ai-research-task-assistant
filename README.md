# AI Research & Task Assistant

A modern, beginner-friendly **multi-agent AI application** built with LangChain, LangGraph, LangSmith-ready tracing, Groq/Qwen, Streamlit, and SQLite.

## Live documentation

Read the complete architecture and feature guide at:

https://sumit1523.github.io/ai-research-task-assistant/

## What the app does

1. Takes a research question.
2. Accepts optional trusted notes, public webpage URLs, PDFs, TXT files, and Markdown files.
3. Runs a visible three-agent handoff:
   - **Researcher Agent** creates an evidence-aware brief.
   - **Critic Agent** checks the brief for gaps, unsupported claims, and uncertainty.
   - **Planner Agent** creates a concise answer and a timed task plan using the reviewed work.
4. Saves sessions and task completion locally in SQLite.
5. Lets the user revise an existing plan, for example: “I only have 30 minutes per day.”

## Architecture

```text
Question + sources
        ↓
LangGraph Coordinator (choose clarification, research, or revision)
        ↓
Researcher Agent → Critic Agent → Planner Agent
        ↓
Summary + task plan → SQLite progress tracking
```

The three roles are independent model calls with distinct prompts and handoffs. The LangGraph Coordinator is deterministic graph logic, not an additional LLM agent.

## How the main technologies are used

| Technology | Role |
| --- | --- |
| LangChain | Text splitting, role-specific prompts, and hosted LLM calls through `ChatGroq`. |
| LangGraph | Shared state and conditional routes across clarification, research, critique, planning, and revisions. |
| LangSmith | Optional tracing for nested graph/agent/model runs and evaluation visibility. |
| Groq + Qwen | Free-tier cloud model provider; no local model download is needed. |
| Streamlit | Modern guided interface, collaboration tabs, task checkboxes, and progress UI. |
| SQLite | Local sessions, task items, and completion state. |

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Copy `.env.example` to `.env`, add a Groq key, then start the app:

```text
GROQ_API_KEY=your_key_here
GROQ_MODEL=qwen/qwen3.6-27b
```

```bash
streamlit run app.py
```

## Optional LangSmith tracing

Add these values to local `.env`:

```text
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=your_langsmith_key_here
LANGSMITH_PROJECT=ai-research-task-assistant
```

After a successful app request, inspect the project in LangSmith to see the Coordinator route and the Researcher, Critic, and Planner calls.

## Test the agent workflow

```bash
.venv/bin/python evaluate.py
```

The evaluation examples check that the graph generates a summary and task plan and preserves expected source concepts. The app has also been exercised through multi-agent handoffs, revisions, PDF handling, task persistence, and Streamlit UI tests.

## Project files

- `app.py` - modern Streamlit interface
- `assistant_graph.py` - LangGraph Coordinator plus Researcher, Critic, and Planner Agents
- `source_loader.py` - URL and upload ingestion
- `storage.py` - SQLite sessions and task tracking
- `evaluate.py` and `evaluation_examples.json` - repeatable quality checks
- `docs/` - GitHub Pages documentation

## Privacy and limits

- `.env`, `.venv`, and `research_tasks.db` are ignored by Git.
- Uploaded source text is read in memory; session data is stored locally.
- The app does not take external actions such as sending emails or making purchases.
- Source labels are helpful grounding context but are not a replacement for human fact-checking.
- The current MVP uses the first six text chunks rather than semantic vector retrieval. Embedding-based retrieval is the next major RAG improvement.
