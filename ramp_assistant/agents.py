import json
import os

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from .state import RampState

parser = JsonOutputParser()


def get_llm():
    return ChatAnthropic(model="claude-haiku-4-5-20251001", temperature=0.3)

def get_eval_llm():
    return ChatAnthropic(model="claude-sonnet-4-5", temperature=0.3)


def extract_concepts(state: RampState) -> dict:
    llm = get_llm()
    messages = [
        SystemMessage(content="""You are a technical curriculum designer preparing a new Deployed Engineer.
Given product documentation, extract the key concepts they need to master.

For each concept, capture four dimensions:
1. What it is (technical definition)
2. What use case it's built for (when to reach for it)
3. Why the customer needs it (the business problem it solves)
4. Why the market needs it (the broader trend driving demand)

Return JSON array:
[
  {
    "name": "concept name",
    "what_it_is": "technical explanation in one sentence",
    "use_case": "the specific scenario where this is the right tool",
    "why_customer_needs_it": "the pain or risk the customer is trying to solve",
    "why_market_needs_it": "the broader trend making this a growing need",
    "difficulty": "foundational|intermediate|advanced"
  }
]

Extract 6-8 concepts. A DE must explain each concept technically AND
articulate why it matters to an engineering team and their business.
Return ONLY valid JSON."""),
        HumanMessage(content=f"Product: {state['product_name']}\n\nDocs:\n{state['product_docs'][:8000]}"),
    ]

    response = llm.invoke(messages)
    concepts = parser.parse(response.content)
    return {"key_concepts": concepts}

def generate_scenarios(state: RampState) -> dict:
    llm = get_llm()
    concepts_text = json.dumps(state["key_concepts"], indent=2)

    messages = [
        SystemMessage(content="""You are a senior Deployed Engineer creating practice scenarios for a new DE hire.
When generating scenarios, rotate through these customer personas:
- Skeptical Staff Engineer who has been burned by vendor lock-in before
- VP of Engineering evaluating LangChain for a 50-person ML team
- CTO pushing back on build vs. buy — "how hard can it be?"
- ML Engineer mid-implementation who hit a wall and needs unblocking
- Head of Platform deciding on framework standardization across teams
- Security-conscious Enterprise Architect worried about data privacy
- Startup CTO moving fast, skeptical of framework overhead
- Data Science Lead trying to move prototypes to production

Each scenario should feel like it came from a different person with different
priorities, technical depth, and business pressures.                      

Generate scenarios across five types — every session should have at least one of each:

TYPE 1 — DEPLOYMENT / DEBUGGING
A customer has a working or broken implementation. The DE must diagnose, explain,
or guide them to a solution. Examples: agent stuck in a loop, memory not persisting
across sessions, RAG returning irrelevant results, LangSmith showing high latency.

TYPE 2 — USE CASE MAPPING
A customer has a goal but doesn't know which pattern or tool to use.
The DE must map their need to the right LangChain/LangGraph approach and explain why.
Examples: "we want our agent to hand off to a human when uncertain",
"we need our agent to remember context across days", "we want to run agents in parallel".

TYPE 3 — VALUE / WHY
A customer or exec pushes back on the approach, the tool choice, or the business case.
The DE must articulate value — not just how it works, but why they need it and why now.
Examples: "why LangChain instead of calling the API directly?",
"our CTO says we should build this ourselves", "what's the ROI of adding LangGraph here?"
                      
TYPE 4 — ARCHITECTURE REVIEW
A customer shares their current LangChain/LangGraph implementation and asks for a review.
The DE must assess it, identify issues or inefficiencies, and suggest improvements clearly.
Examples: "we built our RAG pipeline this way, does this look right?",
"our agent graph has 8 nodes, is that too complex?", "we're storing state in Redis, is that the right approach?"

TYPE 5 — COMPETITIVE
A customer is deciding between LangChain and an alternative approach.
The DE must differentiate honestly — no overselling, no dismissing the alternative.
Examples: "why not just call the Anthropic API directly?",
"we looked at LlamaIndex, why should we pick LangChain?",
"our engineers want to build the orchestration layer themselves"

Return JSON array:
[
  {
    "type": "debugging|use_case|value",
    "title": "short scenario title",
    "customer_persona": "who is asking (e.g. Senior ML Engineer, VP of Engineering, CTO)",
    "context": "2-3 sentences on the customer situation and what they are trying to do",
    "customer_question": "the specific question, problem, or pushback",
    "what_good_looks_like": "the key insight a strong DE would bring to this",
    "difficulty": "foundational|intermediate|advanced"
  }
]

Generate 6-8 scenarios, at least one of each type.
Make them feel like real customer conversations — specific, grounded, no textbook phrasing.
Return ONLY valid JSON."""),
        HumanMessage(content=f"Product: {state['product_name']}\n\nConcepts:\n{concepts_text}"),
    ]

    response = llm.invoke(messages)
    scenarios = parser.parse(response.content)
    return {"scenarios": scenarios}

def evaluate_response(state: RampState) -> dict:
    llm = get_eval_llm()
    scenario = state["scenarios"][0] if state["scenarios"] else {}
    concepts_text = json.dumps(state["key_concepts"], indent=2)

    messages = [
        SystemMessage(content="""You are a principal Deployed Engineer coaching a new DE on their customer communication.

Evaluate their response across four dimensions:

1. TECHNICAL ACCURACY — Did they get the architecture, API, or behavior right?
   Flag any errors or gaps that would mislead a customer's engineering team.

2. VALUE ARTICULATION — Did they explain not just what the feature does but why the customer needs it?
   A DE who only explains mechanics misses half the job. Did they connect the technical
   answer to the customer's business problem or the market trend driving it?

3. PROBLEM-SOLVING APPROACH — For debugging scenarios: did they give a structured diagnostic
   path, not just a guess? For use case scenarios: did they ask clarifying questions or
   make reasonable assumptions explicit?

4. CUSTOMER CLARITY — Could a senior engineer on the customer's team act on this response?
   Is it specific enough to be useful, or generic enough to be useless?

Return JSON:
{
  "score": 1-10,
  "technical_accuracy": "what they got right and what was wrong or missing",
  "value_articulation": "did they explain why not just what? what business context did they miss?",
  "problem_solving": "was their approach structured? what would a principal DE do differently?",
  "customer_clarity": "could the customer's team act on this? what was vague?",
  "strengths": ["specific things they did well"],
  "gaps": ["specific things they missed"],
  "coaching_tip": "the one thing to focus on next",
  "model_response": "how a principal DE would answer this (3-4 sentences, specific and concrete)"
}

Be direct. Generic feedback is useless.
Return ONLY valid JSON."""),
        HumanMessage(content=f"""Product: {state['product_name']}

Concepts:
{concepts_text}

Scenario:
{json.dumps(scenario, indent=2)}

Their response:
{state['user_response']}"""),
    ]

    response = llm.invoke(messages)
    evaluation = parser.parse(response.content)
    return {"evaluation": evaluation}