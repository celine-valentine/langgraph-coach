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