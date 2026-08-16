"""Critic agent: verifies thematic fit (LLM), and drives graduated, alternating constraint relaxation."""
from pydantic import BaseModel, Field

# ---- relaxation caps ----
RUNTIME_STEP = 20          # minutes per runtime relaxation step
RUNTIME_MAX_STEPS = 3      # 3 x 20 = up to +60 min (e.g. 120 -> 180)
YEAR_STEP = 5              # years per side per year relaxation step
YEAR_MAX_STEPS = 4         # 4 x 5 = up to +/-20 years

def _relaxed_to_cap(relax_state):
    """True if any relaxable axis was pushed to its cap (=> results are a 'stretch')."""
    if not relax_state:
        return False
    return (relax_state.get("runtime_steps", 0) >= RUNTIME_MAX_STEPS or
            relax_state.get("year_steps", 0) >= YEAR_MAX_STEPS)

class Verdict(BaseModel):
    movieId: int
    keep: bool = Field(description="true if this movie's plot/tone matches the desired mood/theme")
    reason: str = Field(description="brief justification (one short phrase)")

class CriticVerdicts(BaseModel):
    verdicts: list[Verdict]


CRITIC_SYSTEM = (
    "You are the critic agent in a movie recommendation system. "
    "Given the user's desired mood/theme and candidate movies with plot overviews, "
    "judge whether EACH candidate genuinely matches the requested mood/theme. "
    "Hard constraints (runtime, genre, year) are ALREADY satisfied — do NOT re-check those. "
    "Judge ONLY thematic and tonal fit. Keep a movie if its plot/tone fits the request; drop it if it clearly does not. "
    "Be discerning but fair — do not drop a candidate that reasonably fits."
)


def _has_runtime(plan):
    return plan.get("runtime_max") is not None or plan.get("runtime_min") is not None

def _has_year(plan):
    return plan.get("year_min") is not None or plan.get("year_max") is not None


def relax_plan(plan, relax_state):
    """Relax ONE present constraint, alternating runtime<->year, each capped independently.
    genre and exclusions are NEVER relaxed. Returns (new_plan, new_relax_state, change_msg_or_None)."""
    rs = dict(relax_state) if relax_state else {"runtime_steps": 0, "year_steps": 0, "next": "runtime"}
    p = dict(plan)

    runtime_avail = _has_runtime(p) and rs["runtime_steps"] < RUNTIME_MAX_STEPS
    year_avail    = _has_year(p)    and rs["year_steps"]    < YEAR_MAX_STEPS

    if not runtime_avail and not year_avail:
        return p, rs, None                          # everything relaxable is exhausted -> caller reports "no match"

    # pick side: honor alternation, but skip a side that's unavailable
    side = rs["next"]
    if side == "runtime" and not runtime_avail: side = "year"
    if side == "year"    and not year_avail:    side = "runtime"

    if side == "runtime":
        if p.get("runtime_max") is not None: p["runtime_max"] += RUNTIME_STEP
        if p.get("runtime_min") is not None: p["runtime_min"] = max(0, p["runtime_min"] - RUNTIME_STEP)
        rs["runtime_steps"] += 1
        rs["next"] = "year"
        msg = f"runtime by {RUNTIME_STEP}min (now max={p.get('runtime_max')}, min={p.get('runtime_min')})"
    else:
        if p.get("year_min") is not None: p["year_min"] -= YEAR_STEP
        if p.get("year_max") is not None: p["year_max"] += YEAR_STEP
        rs["year_steps"] += 1
        rs["next"] = "runtime"
        msg = f"year window by +/-{YEAR_STEP} (now {p.get('year_min')}-{p.get('year_max')})"
    return p, rs, msg


class Critic:
    def __init__(self, llm, min_verified=5):
        self.llm = llm
        self.min_verified = min_verified

    def verify(self, plan, candidates):
        if not candidates:
            return [], []
        listing = "\n".join(
            f"- id={c['movieId']} | {c['title']} ({c.get('year')}) | {(c.get('overview') or '')[:200]}"
            for c in candidates)
        prompt = (f"Desired mood/theme: {plan['mood']}\n"
                  f"Semantic intent: {plan['semantic_query']}\n\n"
                  f"Candidates:\n{listing}\n\n"
                  f"For EACH candidate id, decide keep or drop (thematic/tonal fit only), with a brief reason.")
        result = self.llm.structured(prompt, CriticVerdicts, system=CRITIC_SYSTEM)
        vmap = {v.movieId: v.keep for v in result.verdicts}
        verified = [c for c in candidates if vmap.get(c["movieId"], True)]
        return verified, result.verdicts


def critic_node(state, critic):
    plan, cands = state["plan"], state["candidates"]
    relax_state = state.get("relax_state") or {"runtime_steps": 0, "year_steps": 0, "next": "runtime"}
    verified, _ = critic.verify(plan, cands)
    it = state["iterations"] + 1
    kept = len(verified)
    log = [f"CRITIC: {kept}/{len(cands)} candidates match the mood (iteration {it})"]

    # enough matches -> accept
    if kept >= critic.min_verified:
        applied = relax_state.get("applied", [])
        if applied and _relaxed_to_cap(relax_state):
            fb = ("STRETCH: No movies truly matched all your constraints. "
                  "Here are the closest matches, found only after loosening: " + "; ".join(applied))
            tag = "STRETCH"
        elif applied:
            fb = "Relaxed constraints to find matches: " + "; ".join(applied)
            tag = f"ACCEPT (relaxed)"
        else:
            fb = ""
            tag = "ACCEPT"
        return {"verified": verified, "iterations": it, "critic_decision": "accept",
                "relax_state": relax_state, "critic_feedback": fb,
                "trajectory": log + [f"CRITIC: {tag} ({kept} verified)"]}

    # not enough -> try to relax one present constraint
    new_plan, new_rs, change = relax_plan(plan, relax_state)
    if change is None:
        # nothing left to relax -> stop, honest "no full match" (keep whatever matched, if any)
        applied = relax_state.get("applied", [])
        return {"verified": verified, "iterations": it, "critic_decision": "accept",
                "relax_state": relax_state,
                "critic_feedback": "NO_FULL_MATCH",
                "trajectory": log + [f"CRITIC: STOP — no movies matched all constraints after relaxing {applied or 'nothing available'}"]}

    new_rs["applied"] = relax_state.get("applied", []) + [change]
    return {"verified": verified, "iterations": it, "critic_decision": "retry",
            "plan": new_plan, "relax_state": new_rs,
            "critic_feedback": f"only {kept} matched; relaxed {change}",
            "trajectory": log + [f"CRITIC: RETRY — relaxed {change}"]}


def route_after_critic(state):
    return "explain" if state["critic_decision"] == "accept" else "retrieve"
