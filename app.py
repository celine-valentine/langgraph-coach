import json
import os

import streamlit as st
from dotenv import load_dotenv

from coach.graph import build_evaluation_graph, build_ramp_graph

load_dotenv()

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

st.set_page_config(page_title="LangGraph Coach", layout="wide")
st.title("LangGraph Coach")
st.caption("Technical scenario practice and coding drills · Built with LangGraph + Claude · Traces in LangSmith")

st.markdown("""
<style>
[data-testid="stSidebar"] {
    min-width: 380px;
    max-width: 380px;
}
[data-testid="stSidebar"] * {
    word-wrap: break-word !important;
    white-space: normal !important;
    overflow-wrap: break-word !important;
}
[data-testid="stSidebar"] input {
    font-size: 13px;
}
[data-testid="stToolbar"],
header[data-testid="stHeader"],
#MainMenu {
    display: none !important;
    visibility: hidden !important;
}
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────
if "result" not in st.session_state:
    st.session_state.result = None
if "selected_scenario" not in st.session_state:
    st.session_state.selected_scenario = None
if "evaluation" not in st.session_state:
    st.session_state.evaluation = None


tab1, tab2 = st.tabs(["Scenario Practice", "Coding Drills"])

with tab1:
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

    if st.session_state.result:
        st.subheader("Key Concepts")
        st.caption("Study these before practicing. A DE must explain each one technically AND articulate why the customer needs it.")

        for i, c in enumerate(st.session_state.result["key_concepts"], 1):
            with st.expander(f"{i}. [{c['difficulty'].upper()}] {c['name']}"):
                st.markdown(f"**What it is:** {c['what_it_is']}")
                st.markdown(f"**Use case:** {c['use_case']}")
                st.markdown(f"**Why the customer needs it:** {c['why_customer_needs_it']}")
                st.markdown(f"**Why the market needs it:** {c['why_market_needs_it']}")

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

with tab2:
    st.subheader("Coding Drills")
    st.caption("Build muscle memory on real LangChain/LangGraph patterns you'll face in a technical interview.")

    if "drill" not in st.session_state:
        st.session_state.drill = None
    if "code_review" not in st.session_state:
        st.session_state.code_review = None

    difficulty = st.selectbox("Difficulty", ["foundational", "intermediate", "advanced"])

    if st.button("Generate Drill"):
        with st.spinner("Generating coding challenge..."):
            from coach.graph import build_drill_graph
            drill_graph = build_drill_graph()
            drill_result = drill_graph.invoke({
                "product_name": "LangChain",
                "difficulty": difficulty,
                "drill": {},
                "code_submission": "",
                "code_review": {},
            })
        st.session_state.drill = drill_result["drill"]
        st.session_state.code_review = None
        st.rerun()

    if st.session_state.drill:
        d = st.session_state.drill
        st.divider()
        st.subheader(f"[{d['difficulty'].upper()}] {d['title']}")

        st.info(
            "**How this works:** Read the objective and requirements below. "
            "Write your solution in the code box. Submit to get reviewed by a principal DE."
        )

        st.markdown(f"**Objective:** {d['objective']}")
        st.markdown(f"**Context:** {d['context']}")

        st.markdown("**Requirements:**")
        for r in d["requirements"]:
            st.markdown(f"- {r}")

        with st.expander("Starter code"):
            st.code(d["starter_code"], language="python")

        with st.expander("Hints (try without first)"):
            for h in d["hints"]:
                st.markdown(f"- {h}")

        st.markdown(f"**What the interviewer looks for:** {d['what_interviewer_looks_for']}")
        st.divider()

        code_submission = st.text_area(
            "Your solution:",
            height=300,
            placeholder="Write your Python code here...",
        )

        if st.button("Submit for code review"):
            if code_submission.strip():
                with st.spinner("Reviewing your code..."):
                    from coach.graph import build_code_review_graph
                    review_graph = build_code_review_graph()
                    review_result = review_graph.invoke({
                        "product_name": "LangChain",
                        "difficulty": difficulty,
                        "drill": d,
                        "code_submission": code_submission,
                        "code_review": {},
                    })
                st.session_state.code_review = review_result["code_review"]
                st.rerun()
            else:
                st.warning("Write your solution first.")

    if st.session_state.code_review:
        r = st.session_state.code_review
        st.divider()
        st.subheader("Code Review")

        col1, col2 = st.columns([1, 3])
        with col1:
            st.metric("Score", f"{r['score']}/10")
        with col2:
            st.markdown(f"**Correctness:** {r['correctness']}")
            st.markdown(f"**LangChain idioms:** {r['langchain_idioms']}")
            st.markdown(f"**Production readiness:** {r['production_readiness']}")
            st.markdown(f"**Senior DE approach:** {r['senior_de_approach']}")

        if r.get("strengths"):
            st.markdown("**Strengths:**")
            for item in r["strengths"]:
                st.markdown(f"- ✓ {item}")

        if r.get("gaps"):
            st.markdown("**Gaps:**")
            for item in r["gaps"]:
                st.markdown(f"- ✗ {item}")

        if r.get("coaching_tip"):
            st.info(f"**Coaching tip:** {r['coaching_tip']}")

        if r.get("model_solution"):
            with st.expander("Model solution — only open after you've tried"):
                st.code(r["model_solution"], language="python")

with st.sidebar:
    st.subheader("Doc Chat")
    st.caption("Ask anything about LangChain, LangGraph, or LangSmith.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])

    if "input_counter" not in st.session_state:
        st.session_state.input_counter = 0

    question_input = st.text_input("Ask a question...", key=f"doc_chat_{st.session_state.input_counter}")
    send = st.button("Send", key="doc_chat_send")
    question = question_input if send and question_input.strip() else None

    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                from coach.drills import chat_with_docs
                answer = chat_with_docs(
                    question=question,
                    chat_history=st.session_state.chat_history[:-1],
                    docs=LANGCHAIN_DOCS,
                )
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        st.session_state.input_counter += 1
        st.rerun()