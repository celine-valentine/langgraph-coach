"""LangGraph orchestration for the ramp assistant."""

from langgraph.graph import END, StateGraph

from .agents import evaluate_response, extract_concepts, generate_scenarios
from .state import RampState


def build_ramp_graph() -> StateGraph:
    """Build the multi-agent ramp assistant graph.

    Flow:
        product_docs → [extract_concepts] → [generate_scenarios] → PAUSE for user input
        user_response → [evaluate_response] → END
    """

    graph = StateGraph(RampState)

    # Add agent nodes
    graph.add_node("extract_concepts", extract_concepts)
    graph.add_node("generate_scenarios", generate_scenarios)
    graph.add_node("evaluate_response", evaluate_response)

    # Define the flow
    graph.set_entry_point("extract_concepts")
    graph.add_edge("extract_concepts", "generate_scenarios")
    graph.add_edge("generate_scenarios", END)

    return graph.compile()


def build_evaluation_graph() -> StateGraph:
    """Build a separate graph for evaluating user responses.

    Called after the user submits their response to a scenario.
    """

    graph = StateGraph(RampState)

    graph.add_node("evaluate_response", evaluate_response)
    graph.set_entry_point("evaluate_response")
    graph.add_edge("evaluate_response", END)

    return graph.compile()
