import streamlit as st
import os

from assistant_graph import research_agent
from source_loader import load_upload, load_url
from storage import initialise_database, recent_sessions, save_session, set_task_completed, task_items

st.set_page_config(page_title="Research & Task Assistant", page_icon="🔎", layout="wide")
initialise_database()

st.markdown(
    """
    <style>
      .stApp { background: #f7f8fc; }
      .block-container { max-width: 1180px; padding-top: 2.2rem; padding-bottom: 4rem; }
      h1 { letter-spacing: -0.045em; }
      .hero { background: linear-gradient(125deg, #172033, #253e88); color: #fff; padding: 2.2rem; border-radius: 22px; margin-bottom: 1.35rem; }
      .hero h1 { color: #fff; margin: 0 0 .45rem; font-size: 2.55rem; }
      .hero p { color: #dce6ff; margin: 0; font-size: 1.06rem; }
      .agent-row { display: flex; flex-wrap: wrap; gap: .65rem; margin: 1rem 0; }
      .agent-chip { border: 1px solid #ccd6f7; background: #fff; color: #27375f; padding: .55rem .8rem; border-radius: 999px; font-weight: 650; font-size: .9rem; }
      .agent-chip span { color: #65728b; font-weight: 500; }
      [data-testid="stSidebar"] { background: #fff; border-right: 1px solid #e5e9f2; }
      [data-testid="stMetric"] { background: #fff; border: 1px solid #e2e7f1; border-radius: 14px; padding: .65rem; }
      .stButton > button { border-radius: 10px; font-weight: 700; }
    </style>
    <div class="hero">
      <h1>Research with a small AI team.</h1>
      <p>Give the team a question and trusted sources. A Researcher investigates, a Critic checks the findings, and a Planner creates an achievable action plan.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Your AI team")
    st.markdown("🔎 **Researcher**\nBuilds an evidence-aware brief.\n\n🛡️ **Critic**\nChecks gaps and unsupported claims.\n\n🗓️ **Planner**\nCreates and revises the task plan.")
    st.divider()
    tracing = os.getenv("LANGSMITH_TRACING", "").lower() == "true"
    st.caption("LangSmith tracing: " + ("enabled" if tracing else "optional — not configured"))
    st.divider()
    st.subheader("Recent projects")
    for question, created_at in recent_sessions():
        st.caption(f"{created_at[:16]} — {question[:45]}")

st.markdown(
    """<div class="agent-row">
      <div class="agent-chip">1. Researcher <span>finds the useful context</span></div>
      <div class="agent-chip">2. Critic <span>reviews quality and gaps</span></div>
      <div class="agent-chip">3. Planner <span>turns findings into tasks</span></div>
    </div>""",
    unsafe_allow_html=True,
)

with st.container(border=True):
    st.subheader("Start a research session")
    question = st.text_input(
        "What would you like to research?",
        placeholder="Example: How can I learn Docker in two weeks?",
    )
    source_notes = st.text_area(
        "Paste notes or trusted text (optional)",
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

if st.button("Ask my AI team", type="primary", use_container_width=True):
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
