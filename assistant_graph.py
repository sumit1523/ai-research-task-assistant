"""The intentionally small AI agent.

LangChain is used for prompts, model calls and text splitting.
LangGraph connects the agent's decisions into a visible workflow.
"""

import os
import re
from typing import Literal, TypedDict

from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from dotenv import load_dotenv

load_dotenv()


class ResearchState(TypedDict, total=False):
    question: str
    source_notes: str
    current_summary: str
    current_tasks: str
    revision_request: str
    context: str
    route: Literal["clarify", "research", "revise"]
    clarification: str
    research_brief: str
    critique: str
    critique_decision: Literal["APPROVED", "NEEDS_REVIEW"]
    summary: str
    tasks: str


def model() -> ChatGroq:
    """Call a free-tier hosted Qwen model; no local model download is required."""
    return ChatGroq(
        model=os.getenv("GROQ_MODEL", "qwen/qwen3.6-27b"),
        temperature=0.2,
        max_tokens=700,
        reasoning_effort="none",
        reasoning_format="hidden",
    )


def decide_next_step(state: ResearchState) -> ResearchState:
    """An agent decision: vague questions go to a helpful clarification response."""
    if state.get("revision_request") and state.get("current_tasks"):
        return {"route": "revise"}
    question = state["question"].strip()
    if len(question.split()) < 4:
        return {"route": "clarify"}
    return {"route": "research"}


def route(state: ResearchState) -> Literal["clarify", "research", "revise"]:
    return state["route"]


def ask_for_clarification(state: ResearchState) -> ResearchState:
    return {
        "clarification": (
            "Could you add a little more detail? For example: your goal, time available, "
            "experience level, or what kind of result you want."
        )
    }


def prepare_context(state: ResearchState) -> ResearchState:
    """A tiny RAG-style preparation step: split notes and retain the most useful excerpts."""
    notes = state.get("source_notes", "").strip()
    if not notes:
        return {"context": "No user-provided sources. Clearly state when advice is general."}
    splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
    chunks = splitter.split_text(notes)
    # The MVP exposes this simple step first. A vector store can replace this line later.
    return {"context": "\n\n--- Source excerpt ---\n".join(chunks[:6])}


def researcher_agent(state: ResearchState) -> ResearchState:
    """Agent 1: turn prepared context into an evidence-aware research brief."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Researcher Agent. Build a concise, evidence-aware brief from the supplied "
                "context. Use [Source: ...] labels when making factual claims. If there is no source context, "
                "say that the findings are general guidance. Do not create a task plan.",
            ),
            (
                "human",
                "Research question: {question}\n\nPrepared source context:\n{context}\n\n"
                "Return a RESEARCH BRIEF with 3-6 concise bullets, followed by a section named "
                "ASSUMPTIONS OR GAPS if information is missing.",
            ),
        ]
    )
    answer = (prompt | model()).invoke({"question": state["question"], "context": state["context"]})
    content = answer.content if isinstance(answer.content, str) else str(answer.content)
    content = re.sub(r"<think>.*?</think>\\s*", "", content, flags=re.DOTALL)
    return {"research_brief": content.strip()}


def critic_agent(state: ResearchState) -> ResearchState:
    """Agent 2: audit the research brief before the planner relies on it."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Critic Agent. Check whether the research brief answers the question, "
                "distinguishes facts from general advice, and exposes important gaps. Be constructive. "
                "You do not write the final plan.",
            ),
            (
                "human",
                "Question: {question}\n\nResearch brief:\n{research_brief}\n\n"
                "Return exactly:\nDECISION: APPROVED or NEEDS_REVIEW\nCRITIQUE:\n2-4 bullets describing "
                "strengths, risks, or missing information.",
            ),
        ]
    )
    answer = (prompt | model()).invoke({"question": state["question"], "research_brief": state["research_brief"]})
    content = answer.content if isinstance(answer.content, str) else str(answer.content)
    content = re.sub(r"<think>.*?</think>\\s*", "", content, flags=re.DOTALL)
    decision = "NEEDS_REVIEW" if "DECISION: NEEDS_REVIEW" in content.upper() else "APPROVED"
    critique = content.partition("CRITIQUE:")[2].strip() or content.strip()
    return {"critique_decision": decision, "critique": critique}


def planner_agent(state: ResearchState) -> ResearchState:
    """Agent 3: convert reviewed research into a practical answer and task plan."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are the Planner Agent. Use the reviewed research brief and critic feedback to make a "
                "practical, honest plan. Do not invent unsupported facts. If the critic found gaps, reflect "
                "them in cautious wording or a research task.",
            ),
            (
                "human",
                "Question: {question}\n\nResearcher brief:\n{research_brief}\n\n"
                "Critic decision: {critique_decision}\nCritic feedback:\n{critique}\n\n"
                "Return exactly these sections:\nSUMMARY:\n3-5 clear bullets.\n\n"
                "TASK PLAN:\nA numbered plan with 4-7 small tasks. Include an estimated time for each.",
            ),
        ]
    )
    answer = (prompt | model()).invoke(
        {
            "question": state["question"],
            "research_brief": state["research_brief"],
            "critique": state["critique"],
            "critique_decision": state["critique_decision"],
        }
    )
    content = answer.content if isinstance(answer.content, str) else str(answer.content)
    content = re.sub(r"<think>.*?</think>\\s*", "", content, flags=re.DOTALL)
    summary, _, tasks = content.partition("TASK PLAN:")
    return {"summary": summary.replace("SUMMARY:", "").strip(), "tasks": tasks.strip()}


def revise_plan(state: ResearchState) -> ResearchState:
    """Planner Agent's human-in-the-loop revision path."""
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "human",
                "You are the Planner Agent revising a research task plan. Keep the answer concise and practical.\n\n"
                "Question: {question}\nCurrent summary: {current_summary}\nCurrent task plan: {current_tasks}\n"
                "User feedback: {revision_request}\n\n"
                "Return exactly these sections:\nSUMMARY:\nA revised 3-5 bullet summary.\n\n"
                "TASK PLAN:\nA revised numbered plan with 4-7 small tasks and time estimates.",
            )
        ]
    )
    answer = (prompt | model()).invoke(state)
    content = answer.content if isinstance(answer.content, str) else str(answer.content)
    content = re.sub(r"<think>.*?</think>\\s*", "", content, flags=re.DOTALL)
    summary, _, tasks = content.partition("TASK PLAN:")
    return {"summary": summary.replace("SUMMARY:", "").strip(), "tasks": tasks.strip()}


def build_graph():
    graph = StateGraph(ResearchState)
    graph.add_node("decide", decide_next_step)
    graph.add_node("clarify", ask_for_clarification)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("researcher", researcher_agent)
    graph.add_node("critic", critic_agent)
    graph.add_node("planner", planner_agent)
    graph.add_node("revise", revise_plan)
    graph.add_edge(START, "decide")
    graph.add_conditional_edges(
        "decide", route, {"clarify": "clarify", "research": "prepare_context", "revise": "revise"}
    )
    graph.add_edge("clarify", END)
    graph.add_edge("prepare_context", "researcher")
    graph.add_edge("researcher", "critic")
    graph.add_edge("critic", "planner")
    graph.add_edge("planner", END)
    graph.add_edge("revise", END)
    return graph.compile()


research_agent = build_graph()
