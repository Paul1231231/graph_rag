import os
from typing_extensions import TypedDict


class State(TypedDict):
    question: str
    context: list[dict]
    answer: str


def get_neo4j_settings() -> dict[str, str]:
    return {
        "uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "username": os.getenv("NEO4J_USERNAME", "neo4j"),
        "password": os.getenv("NEO4J_PASSWORD", "password"),
        "database": os.getenv("NEO4J_DATABASE", "neo4j"),
    }


def default_answer(question: str) -> str:
    return (
        f"No BOM context was found for: '{question}'. "
        "Try including a part number, assembly number, or supplier name."
    )
