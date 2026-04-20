from typing import TypedDict


class RampState(TypedDict):
    product_docs: str
    product_name: str
    key_concepts: list[dict]
    scenarios: list[dict]
    user_response: str
    evaluation: dict


class DrillState(TypedDict):
    product_name: str
    difficulty: str
    drill: dict
    code_submission: str
    code_review: dict