import json
import os
from dotenv import load_dotenv
from coach.graph import build_ramp_graph, build_evaluation_graph

load_dotenv()

print("KEY:", os.getenv("ANTHROPIC_API_KEY"))

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


def print_header(text):
    print(f"\n{'='*60}")
    print(f"  {text}")
    print(f"{'='*60}\n")


def main():
    print_header("LANGGRAPH COACH — Technical Scenario Practice")
    print("  Built with LangChain + LangGraph + Claude")
    print("  Traces visible in LangSmith\n")

    product_name = input("  Product (press Enter for 'LangChain'): ").strip() or "LangChain"
    custom_docs = input("  Paste docs (press Enter to use built-in LangChain docs): ").strip()
    product_docs = custom_docs if custom_docs else LANGCHAIN_DOCS

    print_header("Extracting concepts and generating scenarios...")

    ramp_graph = build_ramp_graph()
    result = ramp_graph.invoke({
        "product_docs": product_docs,
        "product_name": product_name,
        "key_concepts": [],
        "scenarios": [],
        "user_response": "",
        "evaluation": {},
    })

    print_header("KEY CONCEPTS")
    for i, c in enumerate(result["key_concepts"], 1):
        print(f"  {i}. [{c['difficulty'].upper()}] {c['name']}")
        print(f"     What it is: {c['what_it_is']}")
        print(f"     Use case: {c['use_case']}")
        print(f"     Why customer needs it: {c['why_customer_needs_it']}")
        print(f"     Why market needs it: {c['why_market_needs_it']}\n")

    print_header("PRACTICE SCENARIOS")
    for i, s in enumerate(result["scenarios"], 1):
        print(f"  {i}. [{s['type'].upper()}] {s['title']} — {s['difficulty'].upper()}")
        print(f"     Customer: {s['customer_persona']}")
        print(f"     {s['context']}")
        print(f"     \"{s['customer_question']}\"\n")

    n = len(result["scenarios"])
    choice = input(f"  Pick a scenario (1-{n}): ").strip()
    try:
        idx = max(0, min(int(choice) - 1, n - 1))
    except ValueError:
        idx = 0

    scenario = result["scenarios"][idx]
    print(f"\n  [{scenario['type'].upper()}] {scenario['title']}")
    print(f"  Customer: {scenario['customer_persona']}")
    print(f"  {scenario['context']}")
    print(f"\n  \"{scenario['customer_question']}\"\n")
    print("  Your response (press Enter twice to submit):\n")

    lines = []
    while True:
        line = input()
        if line == "":
            break
        lines.append(line)

    print_header("Evaluating your response...")

    eval_graph = build_evaluation_graph()
    eval_result = eval_graph.invoke({
        **result,
        "scenarios": [scenario],
        "user_response": "\n".join(lines),
        "evaluation": {},
    })

    e = eval_result["evaluation"]
    print_header("COACHING FEEDBACK")
    print(f"  Score: {e['score']}/10\n")
    print(f"  Technical accuracy:  {e['technical_accuracy']}")
    print(f"  Value articulation:  {e['value_articulation']}")
    print(f"  Problem solving:     {e['problem_solving']}")
    print(f"  Customer clarity:    {e['customer_clarity']}")

    if e.get("strengths"):
        print(f"\n  Strengths:")
        for s in e["strengths"]:
            print(f"    + {s}")

    if e.get("gaps"):
        print(f"\n  Gaps:")
        for g in e["gaps"]:
            print(f"    - {g}")

    if e.get("coaching_tip"):
        print(f"\n  Coaching tip: {e['coaching_tip']}")

    if e.get("model_response"):
        print(f"\n  Principal DE would say:")
        print(f"  \"{e['model_response']}\"")

    print(f"\n{'='*60}")
    print("  Done. Check LangSmith for full traces.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()