import streamlit as st
import os

from assistant_graph import research_agent
from source_loader import load_upload, load_url
from storage import initialise_database, recent_sessions, save_session, set_task_completed, task_items

st.set_page_config(page_title="Research & Task Assistant", page_icon="🔎", layout="wide")
initialise_database()

def apply_theme(dark_mode: bool) -> None:
    colors = (
        {"bg": "#111827", "panel": "#1f2937", "text": "#f8fafc", "muted": "#cbd5e1", "line": "#374151", "accent": "#8b9cff"}
        if dark_mode
        else {"bg": "#f7f8fc", "panel": "#ffffff", "text": "#172033", "muted": "#62708a", "line": "#dfe5ef", "accent": "#355bd8"}
    )
    st.markdown(
        f"""
        <style>
          [data-testid="stAppViewContainer"] {{ background: {colors['bg']}; }}
          [data-testid="stHeader"] {{ background: transparent; }}
          .main .block-container {{ max-width: 1160px !important; width: 100% !important; padding: 2rem 2rem 4rem !important; }}
          [data-testid="stSidebar"] {{ background: {colors['panel']}; border-right: 1px solid {colors['line']}; }}
          [data-testid="stSidebar"] * {{ color: {colors['text']}; }}
          h1, h2, h3, p, label, [data-testid="stMarkdownContainer"] {{ color: {colors['text']}; }}
          [data-testid="stCaptionContainer"], [data-testid="stCaptionContainer"] * {{ color: {colors['muted']} !important; }}
          [data-testid="stMetric"] {{ background: {colors['panel']}; border: 1px solid {colors['line']}; border-radius: 14px; padding: .7rem; }}
          [data-testid="stMetric"] * {{ color: {colors['text']}; }}
          [data-testid="stVerticalBlockBorderWrapper"] {{ background: {colors['panel']}; border-color: {colors['line']}; border-radius: 16px; }}
          .stTextInput input, .stTextArea textarea {{ background: {colors['panel']}; color: {colors['text']}; border-color: {colors['line']}; }}
          .stTextInput input::placeholder, .stTextArea textarea::placeholder {{ color: {colors['muted']}; }}
          button {{ border-radius: 10px !important; font-weight: 700 !important; }}
          [data-baseweb="tab-list"] {{ gap: .4rem; }}
          [data-baseweb="tab"] {{ border-radius: 9px; padding: .45rem .8rem; color: {colors['muted']}; }}
          [aria-selected="true"] {{ background: {colors['panel']}; color: {colors['accent']} !important; }}
          @media (max-width: 700px) {{ .main .block-container {{ padding: 1.2rem 1rem 3rem !important; }} }}
        </style>
        """,
        unsafe_allow_html=True,
    )

with st.sidebar:
    dark_mode = st.toggle("🌙 Dark mode", key="dark_mode_toggle")
    st.divider()
    st.header("Your AI team")
    st.markdown("🔎 **Researcher**\nBuilds an evidence-aware brief.\n\n🛡️ **Critic**\nChecks gaps and unsupported claims.\n\n🗓️ **Planner**\nCreates and revises the task plan.")
    st.divider()
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    st.caption("LangSmith tracing: " + ("enabled" if tracing else "optional — not configured"))
    st.divider()
    st.subheader("Recent projects")
    for question, created_at in recent_sessions():
        st.caption(f"{created_at[:16]} — {question[:45]}")

apply_theme(dark_mode)
st.title("Research & Task Assistant")
st.caption("A simple workspace where a Researcher, Critic, and Planner turn sources into an actionable plan.")
agent_one, agent_two, agent_three = st.columns(3)
agent_one.metric("🔎 Researcher", "Find evidence")
agent_two.metric("🛡️ Critic", "Check quality")
agent_three.metric("🗓️ Planner", "Make a plan")
st.write("")

with st.container(border=True):
    st.subheader("What do you want to work on?")
    st.caption("Start with a question. Add sources only when you have useful context to share.")
    question = st.text_input(
        "What would you like to research?",
        placeholder="Example: How can I learn Docker in two weeks?",
    )
    source_notes = st.text_area(
        "Notes or trusted text (optional)",
        placeholder="Paste article notes, meeting notes, or course material. The team will use it as evidence.",
        height=145,
    )
    first, second = st.columns(2)
    with first:
        source_urls = st.text_area(
            "Public web URLs (optional, one per line)", placeholder="https://example.com/article", height=90
        )
    with second:
        uploads = st.file_uploader(
            "Research files (optional)", type=["pdf", "txt", "md"], accept_multiple_files=True
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
            with st.status("Your AI team is working...", expanded=True) as status:
                st.write("🔎 **Researcher:** reading the question and source context")
                st.write("🛡️ **Critic:** checking the research brief for gaps")
                st.write("🗓️ **Planner:** creating a realistic action plan")
                result = research_agent.invoke({"question": question, "source_notes": combined_notes})
                status.update(label="Research session complete", state="complete", expanded=False)
            if result.get("clarification"):
                st.info(result["clarification"])
            else:
                st.success("Your Researcher, Critic, and Planner completed their handoff.")
                session_id = save_session(question, result["summary"], result["tasks"])
                st.session_state.current_research = {
                    "question": question,
                    "summary": result["summary"],
                    "tasks": result["tasks"],
                    "session_id": session_id,
                    "research_brief": result.get("research_brief", ""),
                    "critique": result.get("critique", ""),
                    "critique_decision": result.get("critique_decision", "APPROVED"),
                    "sources": [name for name, _ in sources],
                }
        except Exception as error:
            st.error("The cloud model could not complete this request.")
            st.caption(f"Details: {error}")

if current := st.session_state.get("current_research"):
    st.divider()
    answer_tab, agents_tab, progress_tab = st.tabs(["📌 Answer & plan", "🤝 Agent collaboration", "✅ Progress & sources"])
    with answer_tab:
        left, right = st.columns(2)
        with left:
            st.subheader("Research summary")
            st.markdown(current["summary"])
        with right:
            st.subheader("Your task plan")
            st.markdown(current["tasks"])
    with agents_tab:
        st.subheader("See how the agents worked together")
        with st.expander("🔎 Researcher brief", expanded=True):
            st.markdown(current.get("research_brief", "Research brief is available for new sessions."))
        critic_label = "✅ Approved" if current.get("critique_decision") == "APPROVED" else "⚠️ Needs review"
        with st.expander(f"🛡️ Critic review - {critic_label}", expanded=True):
            st.markdown(current.get("critique", "Critic review is available for new sessions."))
        st.info("🗓️ The Planner used this reviewed brief and critique to create the task plan in the first tab.")
    with progress_tab:
        st.caption("Sources used: " + (", ".join(current.get("sources", [])) or "general guidance"))
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
    st.subheader("Need a different plan?")
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
            st.rerun()

st.divider()
st.caption("Built with LangChain, LangGraph, Groq/Qwen, SQLite, Streamlit, and optional LangSmith tracing.")
