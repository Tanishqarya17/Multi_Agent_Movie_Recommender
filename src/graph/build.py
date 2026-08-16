"""Assembles the four agents into the full LangGraph state machine."""
from langgraph.graph import StateGraph, START, END

from graph.state import AgentState
from agents.planner import Planner, planner_node
from agents.retriever import Retriever, retriever_node
from agents.critic import Critic, critic_node, route_after_critic
from agents.explainer import Explainer, explainer_node


def build_graph(llm, tool_a, tool_b, tool_c, tool_d, reranker, catalog_path,
                min_verified=5, max_iterations=2, top_n=5):
    planner   = Planner(llm)
    retriever = Retriever(tool_a, tool_b, tool_c, tool_d, reranker, catalog_path)
    critic    = Critic(llm, min_verified=min_verified)
    explainer = Explainer(llm, top_n=top_n)

    g = StateGraph(AgentState)
    # bind each agent to its node via a closure (node sees only `state`; agent captured from scope)
    g.add_node("planner",   lambda s: planner_node(s, planner))
    g.add_node("retriever", lambda s: retriever_node(s, retriever))
    g.add_node("critic",    lambda s: critic_node(s, critic))
    g.add_node("explainer", lambda s: explainer_node(s, explainer))

    g.add_edge(START, "planner")
    g.add_edge("planner", "retriever")
    g.add_edge("retriever", "critic")
    g.add_conditional_edges("critic", route_after_critic,
                            {"retrieve": "retriever", "explain": "explainer"})   # critic's decision routes
    g.add_edge("explainer", END)
    return g.compile()
