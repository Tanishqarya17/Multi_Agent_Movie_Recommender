"""Shared state for the multi-agent movie recommendation graph."""
from typing import TypedDict, Annotated
import operator


class AgentState(TypedDict):
    query: str
    plan: dict
    candidates: list
    critic_feedback: str
    critic_decision: str
    relax_state: dict                           # <-- NEW: tracks graduated relaxation progress across passes
    verified: list
    recommendations: list
    iterations: int
    trajectory: Annotated[list, operator.add]
