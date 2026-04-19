import os
from dotenv import load_dotenv
load_dotenv()

from dotenv import load_dotenv
from ramp_assistant.graph import build_ramp_graph

load_dotenv()

graph = build_ramp_graph()

result = graph.invoke({
    "product_docs": "LangGraph is a library for building stateful multi-agent systems using LangChain. It models agent workflows as graphs where nodes are functions and edges define execution flow.",
    "product_name": "LangGraph",
    "key_concepts": [],
    "scenarios": [],
    "user_response": "",
    "evaluation": {},
})

print("CONCEPTS:", result["key_concepts"])
print("SCENARIOS:", result["scenarios"])