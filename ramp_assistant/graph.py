from langgraph.graph import END, StateGraph

from .agents import extract_concepts, generate_scenarios
from .state import RampState


def build_ramp_graph():
    graph = StateGraph(RampState)

    graph.add_node("extract_concepts", extract_concepts)
    graph.add_node("generate_scenarios", generate_scenarios)

    graph.set_entry_point("extract_concepts")
    graph.add_edge("extract_concepts", "generate_scenarios")
    graph.add_edge("generate_scenarios", END)

    return graph.compile()