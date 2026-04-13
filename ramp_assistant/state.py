"""Shared state for the ramp assistant graph."""

from typing import TypedDict


class RampState(TypedDict):
    """State passed between agents in the ramp assistant graph."""

    # Input
    product_docs: str  # Raw product documentation text
    product_name: str  # Name of the product (e.g., "HashiCorp Vault")

    # Agent 1 output: extracted concepts
    key_concepts: list[dict]  # [{name, description, difficulty}]

    # Agent 2 output: practice scenarios
    scenarios: list[dict]  # [{title, context, task, expected_approach}]

    # User interaction
    user_response: str  # User's answer to a scenario

    # Agent 3 output: evaluation
    evaluation: dict  # {score, strengths, gaps, next_steps}
