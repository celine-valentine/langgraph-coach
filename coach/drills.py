import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from .state import DrillState

parser = JsonOutputParser()


def get_llm():
    return ChatAnthropic(model="claude-sonnet-4-5", temperature=0.3)


def generate_drill(state: DrillState) -> dict:
    llm = get_llm()

    messages = [
        SystemMessage(content="""You are a senior Deployed Engineer creating coding drills
for a new DE hire preparing for a technical interview.

Generate a coding challenge using LangChain and LangGraph that matches the difficulty level.

FOUNDATIONAL challenges:
- Build a single LangGraph node from scratch
- Create a TypedDict state schema
- Initialize a ChatAnthropic model and invoke it with messages
- Wire up a JsonOutputParser to an LLM call

INTERMEDIATE challenges:
- Build a 2-3 node sequential graph
- Add conditional edges based on state values
- Add a tool to an agent and handle tool calls
- Add memory that persists across a conversation

ADVANCED challenges:
- Add checkpointing to a graph for human-in-the-loop
- Build parallel node execution
- Add streaming output to a graph
- Build a multi-agent system where agents hand off to each other

Return JSON:
{
  "title": "short drill title",
  "difficulty": "foundational|intermediate|advanced",
  "objective": "one sentence — what the candidate must build",
  "context": "2-3 sentences explaining the scenario and why this matters for a DE",
  "requirements": ["specific requirement 1", "specific requirement 2", "specific requirement 3"],
  "starter_code": "minimal Python skeleton with imports and TODO comments",
  "hints": ["hint 1 if they get stuck", "hint 2"],
  "what_interviewer_looks_for": "what a senior DE would expect to see in a strong solution"
}

Return ONLY valid JSON."""),
        HumanMessage(content=f"Product: {state['product_name']}\nDifficulty: {state['difficulty']}"),
    ]

    response = llm.invoke(messages)
    drill = parser.parse(response.content)
    return {"drill": drill}


def review_code(state: DrillState) -> dict:
    llm = get_llm()
    drill = state["drill"]

    messages = [
        SystemMessage(content="""You are a principal Deployed Engineer reviewing a candidate's
code submission during a technical interview.

Evaluate their code across four dimensions:

1. CORRECTNESS — Does it actually solve the problem? Would it run?
   Flag any bugs, missing imports, or logic errors.

2. LANGCHAIN IDIOMS — Are they using LangChain/LangGraph the right way?
   Flag any anti-patterns, unnecessary complexity, or missed built-in features.

3. PRODUCTION READINESS — Would this hold up in a real customer deployment?
   Look for missing error handling, hardcoded values, or scaling issues.

4. SENIOR DE APPROACH — What would a principal DE do differently?
   Not just correctness — elegance, clarity, and maintainability.

Return JSON:
{
  "score": 1-10,
  "correctness": "does it solve the problem? any bugs?",
  "langchain_idioms": "are they using the framework correctly?",
  "production_readiness": "would this hold up in production?",
  "senior_de_approach": "what would a principal DE do differently?",
  "strengths": ["specific things done well"],
  "gaps": ["specific issues to fix"],
  "coaching_tip": "the one most important thing to improve",
  "model_solution": "a clean 10-15 line example showing the right approach"
}

Be direct. This is a technical interview — vague feedback wastes everyone's time.
Return ONLY valid JSON."""),
        HumanMessage(content=f"""Drill: {json.dumps(drill, indent=2)}

Candidate's code:
{state['code_submission']}"""),
    ]

    response = llm.invoke(messages)
    code_review = parser.parse(response.content)
    return {"code_review": code_review}

def chat_with_docs(question: str, chat_history: list, docs: str) -> str:
    llm = get_llm()

    history_text = ""
    for msg in chat_history:
        role = "You" if msg["role"] == "user" else "Assistant"
        history_text += f"{role}: {msg['content']}\n"

    messages = [
        SystemMessage(content=f"""You are a LangChain technical expert helping a Deployed Engineer
candidate prepare for their interview.

Answer questions about LangChain, LangGraph, and LangSmith clearly and concisely.
Be technical and specific — this person is preparing for a DE role, not a beginner tutorial.
When relevant, include short code examples.

Reference documentation:
{docs}

Previous conversation:
{history_text}"""),
        HumanMessage(content=question),
    ]

    response = llm.invoke(messages)
    return response.content