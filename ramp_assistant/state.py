from typing import TypedDict


class RampState(TypedDict):
    product_docs: str
    product_name: str
    key_concepts: list[dict]
    scenarios: list[dict]
    user_response: str
    evaluation: dict