# LangGraph Coach

A multi-agent AI system that coaches you through technical scenarios and coding drills for AI-native products.

Built with LangGraph, Claude, and LangSmith. Runs locally in your browser via Streamlit.

---

## Demo

<!-- After recording: replace VIDEO_ID with your YouTube video ID -->
<!-- Format: [![Demo](https://img.youtube.com/vi/VIDEO_ID/maxresdefault.jpg)](https://www.youtube.com/watch?v=VIDEO_ID) -->

*Demo video coming soon*

---

## What this teaches you

By reading the code and running this app, you'll understand:

- **How to structure a multi-agent LangGraph system** — 4 independent StateGraphs, each with isolated TypedDict state, composable without coupling
- **Why state isolation matters** — RampState and DrillState are separate schemas; the ramp graph can't accidentally touch the drill graph's state
- **How to chain agents with clean handoffs** — concept extraction → scenario generation → evaluation, each node returning typed updates to shared state
- **LangSmith tracing in practice** — every agent run is observable: prompts, outputs, token counts, per-node latency, all without changing your application code
- **Session state management in Streamlit** — how to build multi-step UIs that don't reset between user actions

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Streamlit UI                         │
│                                                         │
│  Tab 1: Scenario Practice    Tab 2: Coding Drills       │
│  ┌─────────────────────┐    ┌──────────────────────┐   │
│  │   Ramp Graph        │    │   Drill Graph        │   │
│  │  extract_concepts   │    │   generate_drill     │   │
│  │       ↓             │    │                      │   │
│  │  generate_scenarios │    │   Code Review Graph  │   │
│  │                     │    │   review_code        │   │
│  │  Evaluation Graph   │    └──────────────────────┘   │
│  │  evaluate_response  │                               │
│  └─────────────────────┘    Sidebar: Doc Chat          │
│                              (persistent chat history) │
└─────────────────────────────────────────────────────────┘
                          │
                    LangSmith
              (traces every agent run)
```

**4 StateGraphs, 2 state schemas:**

| Graph | Nodes | State |
|-------|-------|-------|
| `build_ramp_graph` | extract_concepts → generate_scenarios | RampState |
| `build_evaluation_graph` | evaluate_response | RampState |
| `build_drill_graph` | generate_drill | DrillState |
| `build_code_review_graph` | review_code | DrillState |

---

## Quick start

```bash
git clone https://github.com/celine-valentine/langgraph-coach
cd langgraph-coach
pip install -r requirements.txt
cp .env.example .env   # add your API keys
streamlit run app.py
```

Open `http://localhost:8501` — the app loads immediately.

---

## What you need

```
ANTHROPIC_API_KEY=        # claude.ai → API keys
LANGCHAIN_API_KEY=        # smith.langchain.com → Settings → API keys
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=ramp-assistant
```

LangSmith has a free tier — the tracing is worth setting up. You'll see every agent run live as you use the app.

---

## How it works

### The ramp graph — two agents, one pipeline

The first time you hit "Generate Concepts & Scenarios," two agents run in sequence:

**Agent 1 — concept extractor** reads the product docs and returns structured objects:
```python
{
  "name": "StateGraph",
  "what_it_is": "A directed graph where nodes are Python functions...",
  "use_case": "Building a multi-step agent that routes based on LLM output",
  "why_customer_needs_it": "...",
  "why_market_needs_it": "...",
  "difficulty": "intermediate"
}
```

**Agent 2 — scenario generator** takes those concepts and builds interview scenarios across 5 types (technical deep-dive, objection handling, architecture design, competitive, implementation) and 8 customer personas (CTO, security architect, ML engineer, etc.) at three difficulty levels.

The key design choice: each agent returns a **partial state update**. LangGraph merges it into shared state automatically. No passing objects between functions manually.

### The evaluation graph — coaching, not just scoring

When you submit a response, the evaluation agent scores across 4 dimensions:

- **Technical accuracy** — did you get the facts right?
- **Value articulation** — did you connect technical detail to business outcome?
- **Problem solving** — did you address the customer's actual situation?
- **Customer clarity** — would a non-engineer follow this?

Then it gives you: specific strengths, specific gaps, a coaching tip, and a model response showing what a principal DE would say. Score/10 is secondary — the coaching text is the product.

### State isolation — why two schemas

`RampState` and `DrillState` are separate TypedDicts with no shared fields:

```python
class RampState(TypedDict):
    product_docs: str
    product_name: str
    key_concepts: list[dict]
    scenarios: list[dict]
    user_response: str
    evaluation: dict

class DrillState(TypedDict):
    product_name: str
    difficulty: str
    drill: dict
    code_submission: str
    code_review: dict
```

The ramp graphs and drill graphs can't access each other's state at all. This isn't just clean code — it's how you prevent agents from reasoning about irrelevant context and producing degraded outputs.

### LangSmith tracing

Set `LANGCHAIN_TRACING_V2=true` and every run appears in your LangSmith dashboard at `smith.langchain.com`. No instrumentation code needed — LangChain and LangGraph trace automatically.

What you see per run: full prompt sent to Claude, full response, token count, latency per node, the state before and after each node. When an agent produces a bad output, you open the trace and immediately see why.

---

## What I'd do differently

**Hardcode less product context.** The LangChain docs are embedded as a string in `app.py`. For a real product, this should be a retriever pulling from chunked docs — not a static blob. The scenario quality degrades when the context window gets crowded.

**Add persona memory across sessions.** Right now, every session starts fresh. A real coaching system should remember how you performed on previous scenarios and increase difficulty on concepts you've already passed.

**Separate the UI from the agent logic more cleanly.** `app.py` does too much — UI rendering, graph invocation, and session state management all in one file. For a production app, the agent invocations belong in a separate service layer.

**Use streaming.** The 30-second spinner while agents run is a bad UX. LangGraph supports streaming state updates — each concept could appear as it's generated instead of waiting for the full graph to finish.

---

## Stack

- **[LangGraph](https://langchain-ai.github.io/langgraph/)** — stateful multi-agent orchestration
- **[LangChain](https://python.langchain.com/)** — LLM abstractions, prompt management
- **[Claude (Anthropic)](https://www.anthropic.com/)** — the LLM powering all agents
- **[LangSmith](https://smith.langchain.com/)** — observability and tracing
- **[Streamlit](https://streamlit.io/)** — UI

---

Built by [Celine Valentine](https://celinevalentine.com) · [LinkedIn](https://linkedin.com/in/celinevalentine)
