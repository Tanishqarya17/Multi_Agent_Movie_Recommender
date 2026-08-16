"""Planner agent: natural-language query -> structured, validated Plan."""
from typing import Optional, Literal
from pydantic import BaseModel, Field

MOVIELENS_GENRES = ["Action", "Adventure", "Animation", "Children", "Comedy", "Crime",
                    "Documentary", "Drama", "Fantasy", "Film-Noir", "Horror", "Musical",
                    "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western", "IMAX"]


class Plan(BaseModel):
    anchor_titles: list[str] = Field(default_factory=list,
        description="Exact movie titles the user named as references/anchors. Empty list if none.")
    mood: str = Field(description="The mood, tone, or themes the user wants, in a few words.")
    semantic_query: str = Field(
        description="A concise search phrase capturing plot/mood/theme for semantic search, "
                    "stripped of hard constraints like runtime or year.")
    strategy: Literal["cf", "semantic", "hybrid"] = Field(
        description="'cf' if the user only named anchor movies; 'semantic' if only a vibe/theme; "
                    "'hybrid' if both.")
    prefer_popular: bool = Field(
        description="True if the user wants well-known/popular films; "
                    "False for hidden gems / underrated / obscure.")
    genres_include: list[str] = Field(default_factory=list,
        description=f"Required genres, chosen ONLY from this list: {MOVIELENS_GENRES}. Empty if none.")
    runtime_max: Optional[int] = Field(default=None, description="Max runtime in minutes, or null.")
    runtime_min: Optional[int] = Field(default=None, description="Min runtime in minutes, or null.")
    year_min: Optional[int] = Field(default=None, description="Earliest release year, or null.")
    year_max: Optional[int] = Field(default=None, description="Latest release year, or null.")
    exclude_titles: list[str] = Field(default_factory=list,
        description="Movie titles the user wants excluded. Empty list if none.")


PLANNER_SYSTEM = (
    "You are the planning agent in a movie recommendation system. "
    "Decompose the user's request into a precise, structured plan for downstream tools. "
    f"For genres, use ONLY these exact labels: {MOVIELENS_GENRES}. "
    "Read 'underrated', 'obscure', 'hidden gem', 'deep cut', 'nothing mainstream' as prefer_popular=False; "
    "read 'popular', 'classic', 'well-known', 'famous', 'mainstream', 'nothing too obscure' as prefer_popular=True. "
    "If the user only names movies -> strategy='cf'; only describes a vibe -> strategy='semantic'; both -> strategy='hybrid'. "
    "IMPORTANT: semantic_query must describe ONLY plot, mood, tone, or themes for plot-based search. "
    "NEVER put runtime, year/decade, popularity words (underrated/obscure/popular/mainstream), or the words 'movie'/'film' in it. "
    "Example: for 'an underrated 80s horror, short' -> semantic_query='scary supernatural horror', NOT 'underrated obscure 80s horror movie'."
)


class Planner:
    def __init__(self, llm):
        self.llm = llm

    def plan(self, query: str) -> Plan:
        p = self.llm.structured(query, Plan, system=PLANNER_SYSTEM)
        valid = set(MOVIELENS_GENRES)
        p.genres_include = [g for g in p.genres_include if g in valid]   # defensive: drop invalid genres
        return p


def planner_node(state, planner: Planner):
    """LangGraph node. (Bound to a Planner instance via functools.partial when the graph is assembled.)"""
    p = planner.plan(state["query"])
    return {
        "plan": p.model_dump(),
        "trajectory": [f"PLANNER: strategy={p.strategy} | anchors={p.anchor_titles} | "
                       f"popular={p.prefer_popular} | genres={p.genres_include} | "
                       f"runtime<=({p.runtime_max}) | years=({p.year_min},{p.year_max}) | "
                       f"exclude={p.exclude_titles}"],
    }
