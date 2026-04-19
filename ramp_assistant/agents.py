import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.output_parsers import JsonOutputParser

from .state import RampState

llm = ChatAnthropic(model="claude-sonnet-4-5", temperature=0.3)
parser = JsonOutputParser()


def extract_concepts(state: RampState) -> dict:
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
    concepts_text = json.dumps(state["key_concepts"], indent=2)

    messages = [
        SystemMessage(content="""You are a senior Deployed Engineer creating practice scenarios for a new DE hire.

Generate scenarios across three types — every session should have at least one of each:

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

Generate 4-5 scenarios, at least one of each type.
Make them feel like real customer conversations — specific, grounded, no textbook phrasing.
Return ONLY valid JSON."""),
        HumanMessage(content=f"Product: {state['product_name']}\n\nConcepts:\n{concepts_text}"),
    ]

    response = llm.invoke(messages)
    scenarios = parser.parse(response.content)
    return {"scenarios": scenarios}