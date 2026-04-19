import json
import os

import streamlit as st
from dotenv import load_dotenv

from ramp_assistant.graph import build_evaluation_graph, build_ramp_graph

load_dotenv(dotenv_path="/Users/dandan/projects/langchain-ramp-assistant/.env")

LANGCHAIN_DOCS = """
LangChain is a framework for building applications powered by language models.
It provides abstractions for chaining LLM calls, managing prompts, parsing outputs,
and connecting to external data and tools.

Key components:
- Chat Models: Wrappers around LLM providers (OpenAI, Anthropic, etc.) with a unified interface
- Prompts: Templates and message formatters that make prompts reusable and composable
- Output Parsers: Convert raw LLM text into structured Python objects (JSON, Pydantic, etc.)
- Chains: Sequences of components piped together using LCEL (LangChain Expression Language)
- Retrievers: Fetch relevant documents from vector stores or other sources for RAG
- Tools: Functions an agent can call to interact with external systems
- Agents: LLM-driven decision loops that choose which tools to call and when

LangGraph is LangChain's library for building stateful multi-agent systems.
Key concepts:
- StateGraph: A graph where nodes are functions and edges define execution flow
- State: A TypedDict shared across all nodes — each node reads from it and returns updates
- Nodes: Python functions that do work on the state
- Edges: Define what runs after each node (static or conditional based on state)
- Checkpointing: Persist state between runs for human-in-the-loop workflows

LangSmith is the observability and evaluation platform for LangChain apps.
Key features:
- Automatic tracing: Every LLM call, chain, and agent run is logged with inputs, outputs, latency, tokens
- Datasets and evaluations: Build ground truth datasets, run evals against them
- Playground: Test prompts and chains interactively
"""

st.set_page_config(page_title="Ramp Assistant", layout="wide")
st.title("Ramp Assistant — DE Interview Prep")
st.caption("Built with LangChain + LangGraph + Claude · Traces in LangSmith")

# ── Session state ──────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "selected_scenario" not in st.session_state:
    st.session_state.selected_scenario = None
if "evaluation" not in st.session_state:
    st.session_state.evaluation = None

# ── Step 1: Product input ──────────────────────────────────────────
with st.expander("Step 1 — Product Setup", expanded=True):
    product_name = st.text_input("Product name", value="LangChain")
    use_default = st.checkbox("Use built-in LangChain docs", value=True)
    custom_docs = ""
    if not use_default:
        custom_docs = st.text_area("Paste product docs", height=200)

    if st.button("Generate Concepts & Scenarios"):
        product_docs = LANGCHAIN_DOCS if use_default else custom_docs
        with st.spinner("Running agents — this takes ~30 seconds..."):
            graph = build_ramp_graph()
            st.session_state.result = graph.invoke({
                "product_docs": product_docs,
                "product_name": product_name,
                "key_concepts": [],
                "scenarios": [],
                "user_response": "",
                "evaluation": {},
            })
        st.session_state.selected_scenario = None
        st.session_state.evaluation = None
        st.success("Done — review the concepts below, then pick a scenario to practice.")

# ── Step 2: Concepts ───────────────────────────────────────────────
if st.session_state.result:
    st.subheader("Key Concepts")
    st.caption("Study these before practicing. A DE must explain each one technically AND articulate why the customer needs it.")

    for i, c in enumerate(st.session_state.result["key_concepts"], 1):
        with st.expander(f"{i}. [{c['difficulty'].upper()}] {c['name']}"):
            st.markdown(f"**What it is:** {c['what_it_is']}")
            st.markdown(f"**Use case:** {c['use_case']}")
            st.markdown(f"**Why the customer needs it:** {c['why_customer_needs_it']}")
            st.markdown(f"**Why the market needs it:** {c['why_market_needs_it']}")

    # ── Step 3: Pick scenario ──────────────────────────────────────
    st.divider()
    st.subheader("Practice Scenarios")
    st.caption("Pick one scenario to practice. Each simulates a real customer conversation a DE would face.")

    scenarios = st.session_state.result["scenarios"]
    for i, s in enumerate(scenarios):
        label = f"{i+1}. [{s['type'].upper()}] {s['title']} — {s['difficulty'].upper()}"
        with st.expander(label):
            st.markdown(f"**Type:** {s['type'].upper()} — {s['customer_persona']}")
            st.markdown(f"**Situation:** {s['context']}")
            st.markdown(f"**Customer says:** \"{s['customer_question']}\"")
            if st.button("Practice this scenario", key=f"pick_{i}"):
                st.session_state.selected_scenario = s
                st.session_state.evaluation = None
                st.rerun()

    # ── Step 4: Respond ────────────────────────────────────────────
    if st.session_state.selected_scenario:
        s = st.session_state.selected_scenario
        st.divider()
        st.subheader(f"Your Turn — [{s['type'].upper()}] {s['title']}")

        st.info(
            "**How this works:** You are the Deployed Engineer on a customer call. "
            "Read the situation and the customer's question below, then respond in the text box "
            "as if you're speaking directly to the customer. Be specific — vague answers score low. "
            "Hit Submit to get coaching feedback from a principal DE."
        )

        st.markdown("**Situation:**")
        st.markdown(f"{s['context']}")
        st.markdown(f"**Customer ({s['customer_persona']}) says:**")
        st.markdown(f"> {s['customer_question']}")
        st.markdown("---")

        user_response = st.text_area(
            "Your response as the DE:",
            height=150,
            placeholder="Speak directly to the customer. Be technical, specific, and clear about the value...",
        )

        if st.button("Submit for coaching"):
            if user_response.strip():
                with st.spinner("Evaluating your response..."):
                    eval_graph = build_evaluation_graph()
                    eval_result = eval_graph.invoke({
                        **st.session_state.result,
                        "scenarios": [s],
                        "user_response": user_response,
                        "evaluation": {},
                    })
                st.session_state.evaluation = eval_result["evaluation"]
                st.rerun()
            else:
                st.warning("Type a response first.")

    # ── Step 5: Feedback ───────────────────────────────────────────
    if st.session_state.evaluation:
        e = st.session_state.evaluation
        st.divider()
        st.subheader("Coaching Feedback")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Score", f"{e['score']}/10")
        with col2:
            st.markdown(f"**Technical accuracy:** {e['technical_accuracy']}")
            st.markdown(f"**Value articulation:** {e['value_articulation']}")
            st.markdown(f"**Problem solving:** {e['problem_solving']}")
            st.markdown(f"**Customer clarity:** {e['customer_clarity']}")

        if e.get("strengths"):
            st.markdown("**Strengths:**")
            for item in e["strengths"]:
                st.markdown(f"- ✓ {item}")

        if e.get("gaps"):
            st.markdown("**Gaps:**")
            for item in e["gaps"]:
                st.markdown(f"- ✗ {item}")

        if e.get("coaching_tip"):
            st.info(f"**Coaching tip:** {e['coaching_tip']}")

        if e.get("model_response"):
            st.success(f"**Principal DE would say:** \"{e['model_response']}\"")
