# AI Research & Task Assistant

A small, local-first learning project built with LangChain, LangGraph, and optional LangSmith.

## What it does

1. Accepts a research question and optional notes/source text.
2. Uses **LangChain** to split the notes into useful context and call a hosted LLM.
3. Uses a **LangGraph agent workflow** to choose whether to ask a clarification, research, or revise a plan from feedback.
4. Saves the result locally in SQLite.
5. Uses **LangSmith** automatically if its environment variables are configured.

## Run it

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a free Groq API key at [console.groq.com/keys](https://console.groq.com/keys), then set it before starting the app:

```bash
export GROQ_API_KEY="paste_your_key_here"
streamlit run app.py
```

The app calls Groq's hosted `qwen/qwen3.6-27b` model—nothing large is downloaded to your computer. Groq's free plan has rate and daily-token limits, which are enough for learning and personal development. If you set `LANGSMITH_TRACING=true` and `LANGSMITH_API_KEY`, LangSmith will trace the graph without changing application code.

## Features now included

- Paste your own notes, add public webpage URLs, or upload `.pdf`, `.txt`, and `.md` sources.
- Receive a concise research summary and timed task plan.
- Give feedback such as “I only have 30 minutes per day” and let the agent revise its plan.
- Save completed runs locally in SQLite.

## Evaluate the agent

```bash
.venv/bin/python evaluate.py
```

The small evaluation dataset checks that the agent returns both required sections and retains an expected source concept. With LangSmith tracing enabled, the evaluation runs appear in the same project alongside normal app runs.

## Project map

- `app.py` — friendly Streamlit interface
- `assistant_graph.py` — LangGraph agent and LangChain prompts
- `storage.py` — local SQLite history
- `source_loader.py` — webpage and user-file ingestion
- `evaluate.py` — small, repeatable agent evaluation
- `research_tasks.db` — created automatically; your saved sessions stay on your computer
