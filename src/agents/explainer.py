"""Explainer agent: verified candidates -> ranked recommendations with grounded, cited explanations."""
from pydantic import BaseModel, Field


class Recommendation(BaseModel):
    movieId: int = Field(description="movieId chosen ONLY from the provided verified candidates")
    explanation: str = Field(
        description="2-3 sentences on why it fits, citing specific plot elements from THIS movie's overview "
                    "and (if anchors are given) its thematic link to them")

class Recommendations(BaseModel):
    ranked: list[Recommendation]


EXPLAINER_SYSTEM = (
    "You are the explainer agent in a movie recommendation system. "
    "From the VERIFIED candidates provided (each with a plot overview), select and rank the best matches "
    "for the user's request, best first. "
    "For each pick, write 2-3 sentences that are STRICTLY GROUNDED in that movie's provided overview — "
    "cite specific plot elements, characters, or themes that actually appear in it. "
    "If anchor movies are named, draw an explicit thematic link to them. "
    "RULES: never invent plot details not in the overview; never recommend a movie not in the provided list."
)


class Explainer:
    def __init__(self, llm, top_n=5):
        self.llm = llm
        self.top_n = top_n

    def explain(self, plan, verified):
        if not verified:
            return []
        listing = "\n".join(
            f"- id={c['movieId']} | {c['title']} ({c.get('year')}) | overview: {(c.get('overview') or '')[:300]}"
            for c in verified)
        anchors = plan.get("anchor_titles") or []
        prompt = (
            f"User wants: {plan['mood']}\n"
            f"Semantic intent: {plan['semantic_query']}\n"
            f"Anchor movies (for thematic linking): {anchors if anchors else 'none'}\n\n"
            f"Verified candidates:\n{listing}\n\n"
            f"Select and rank the top {self.top_n}. Give each a grounded 2-3 sentence explanation.")
        result = self.llm.structured(prompt, Recommendations, system=EXPLAINER_SYSTEM)

        by_id = {c['movieId']: c for c in verified}
        recs = []
        for r in result.ranked:
            if r.movieId in by_id:                       # defensive: only real verified movies survive
                c = by_id[r.movieId]
                recs.append({"movieId": r.movieId, "title": c['title'], "year": c.get('year'),
                             "genres": c.get('genres'), "explanation": r.explanation})
        return recs[:self.top_n]


def explainer_node(state, explainer):
    recs = explainer.explain(state["plan"], state["verified"])
    return {"recommendations": recs,
            "trajectory": [f"EXPLAINER: produced {len(recs)} grounded recommendations"]}
