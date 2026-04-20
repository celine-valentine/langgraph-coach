from langgraph.graph import END, StateGraph

from .agents import extract_concepts, generate_scenarios, evaluate_response
from .state import RampState, DrillState

def build_ramp_graph():
    graph = StateGraph(RampState)

    graph.add_node("extract_concepts", extract_concepts)
    graph.add_node("generate_scenarios", generate_scenarios)

    graph.set_entry_point("extract_concepts")
    graph.add_edge("extract_concepts", "generate_scenarios")
    graph.add_edge("generate_scenarios", END)

    return graph.compile()

def build_evaluation_graph():
    graph = StateGraph(RampState)

    graph.add_node("evaluate_response", evaluate_response)
    graph.set_entry_point("evaluate_response")
    graph.add_edge("evaluate_response", END)

    return graph.compile()

from .drills import generate_drill, review_code
from .state import DrillState


def build_drill_graph():
    graph = StateGraph(DrillState)

    graph.add_node("generate_drill", generate_drill)
    graph.set_entry_point("generate_drill")
    graph.add_edge("generate_drill", END)

    return graph.compile()


def build_code_review_graph():
    graph = StateGraph(DrillState)

    graph.add_node("review_code", review_code)
    graph.set_entry_point("review_code")
    graph.add_edge("review_code", END)

    return graph.compile()