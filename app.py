import streamlit as st
import os

from assistant_graph import research_agent
from source_loader import load_upload, load_url
from storage import initialise_database, recent_sessions, save_session, set_task_completed, task_items

st.set_page_config(page_title="Research & Task Assistant", page_icon="🔎", layout="wide")
initialise_database()

st.title("🔎 Research & Task Assistant")
st.caption("Turn a question and your notes into a clear answer and an achievable plan.")

with st.sidebar:
    st.header("How it works")
    st.markdown("1. Ask a question\n2. Add optional source notes\n3. Review your summary and tasks")
    st.divider()
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    st.caption("LangSmith tracing: " + ("enabled" if tracing else "optional — not configured"))
    st.divider()
    st.subheader("Recent projects")
    for question, created_at in recent_sessions():
        st.caption(f"{created_at[:16]} — {question[:45]}")

question = st.text_input(
    "What would you like to research?",
    placeholder="Example: How can I learn Docker in two weeks?",
)
source_notes = st.text_area(
    "Optional: paste notes, article text, or course material",
    placeholder="Paste trusted information here. The assistant will use it as research context.",
    height=180,
)
source_urls = st.text_area(
    "Optional: add public web URLs (one per line)",
    placeholder="https://example.com/article",
    height=90,
)
uploads = st.file_uploader(
    "Optional: upload research files", type=["pdf", "txt", "md"], accept_multiple_files=True
)

if st.button("Create my research plan", type="primary", use_container_width=True):
    if not question.strip():
        st.warning("Please enter a research question first.")
    else:
        try:
            sources = []
            combined_notes = source_notes.strip()
            if combined_notes:
                sources.append(("Your notes", combined_notes))
            for url in filter(None, (line.strip() for line in source_urls.splitlines())):
                sources.append(load_url(url))
            for upload in uploads or []:
                sources.append(load_upload(upload))
            if sources:
                combined_notes = "\n\n".join(f"[Source: {name}]\n{text}" for name, text in sources)
        except Exception as error:
            st.error(f"Source issue: {error}")
            st.stop()
        try:
            with st.spinner("The agent is researching and creating your plan..."):
                result = research_agent.invoke({"question": question, "source_notes": combined_notes})
            if result.get("clarification"):
                st.info(result["clarification"])
            else:
                left, right = st.columns(2)
                with left:
                    st.subheader("Research summary")
                    st.markdown(result["summary"])
                with right:
                    st.subheader("Your task plan")
                    st.markdown(result["tasks"])
                st.caption("Sources used: " + (", ".join(name for name, _ in sources) if sources else "general guidance"))
                session_id = save_session(question, result["summary"], result["tasks"])
                st.session_state.current_research = {
                    "question": question,
                    "summary": result["summary"],
                    "tasks": result["tasks"],
                    "session_id": session_id,
                }
                st.success("Saved locally to your project history.")
        except Exception as error:
            st.error("The cloud model could not complete this request.")
            st.caption(f"Details: {error}")

if current := st.session_state.get("current_research"):
    st.divider()
    st.subheader("Track your progress")
    items = task_items(current["session_id"])
    for task_id, task_text, completed in items:
        is_complete = st.checkbox(task_text, value=completed, key=f"task_{task_id}")
        if is_complete != completed:
            set_task_completed(task_id, is_complete)
    if items:
        complete_count = sum(completed for _, _, completed in items)
        st.progress(complete_count / len(items), text=f"{complete_count} of {len(items)} tasks completed")

    st.divider()
    st.subheader("Revise this plan")
    feedback = st.text_input("What should change?", placeholder="Example: I have only 30 minutes a day.")
    if st.button("Revise my plan"):
        if not feedback.strip():
            st.warning("Tell the agent what you would like to change.")
        else:
            with st.spinner("Revising your plan..."):
                revised = research_agent.invoke(
                    {
                        "question": current["question"],
                        "current_summary": current["summary"],
                        "current_tasks": current["tasks"],
                        "revision_request": feedback,
                    }
                )
            current.update(summary=revised["summary"], tasks=revised["tasks"])
            current["session_id"] = save_session(current["question"], revised["summary"], revised["tasks"])
            st.success("Your revised plan is ready.")
            st.markdown(revised["summary"])
            st.markdown(revised["tasks"])

st.divider()
st.subheader("What you are learning")
st.markdown("**LangChain** prepares source text and calls the model. **LangGraph** controls the decision path. **LangSmith** can trace each run when enabled in `.env`.")
