"""Retriever agent: executes the plan via Tools A/B/C/D + re-ranker. Plan-driven, no LLM call."""
import re
import pandas as pd


class Retriever:
    def __init__(self, tool_a, tool_b, tool_c, tool_d, reranker, catalog_path,
                 cf_n=150, sem_n=50, sem_retrieve_n=400, pop_weight=0.15, candidate_cap=25):
        
        self.a, self.b, self.c, self.d, self.rr = tool_a, tool_b, tool_c, tool_d, reranker
        self.cf_n, self.sem_n, self.pop_weight, self.cap = cf_n, sem_n, pop_weight, candidate_cap
        self.sem_retrieve_n = sem_retrieve_n                    
        cat = pd.read_parquet(catalog_path)[["movieId", "title", "rating_count"]].copy()
        cat["norm"] = cat["title"].apply(self._norm)
        self._cat = cat

    @staticmethod
    def _norm(t):
        t = str(t).lower().strip()
        t = re.sub(r"\s*\(\d{4}\)\s*$", "", t)
        t = re.sub(r"^(the|a|an)\s+", "", t)
        return t.strip()

    def resolve_title(self, title):
        n = self._norm(title)
        hits = self._cat[self._cat["norm"] == n]
        if len(hits) == 0:
            hits = self._cat[self._cat["norm"].str.contains(re.escape(n), na=False)]
        if len(hits) == 0:
            return None
        return int(hits.sort_values("rating_count", ascending=False).iloc[0]["movieId"])

    def retrieve(self, plan):
        pool, log = {}, []

        # 1) CF via anchor fold-in
        if plan["strategy"] in ("cf", "hybrid") and plan["anchor_titles"]:
            anchor_ids = [mid for t in plan["anchor_titles"] if (mid := self.resolve_title(t))]
            if anchor_ids:
                cf = self.a.candidates_from_anchors(anchor_ids, n=self.cf_n)
                for mid in cf["candidates"]:
                    pool[mid] = {"source": "cf", "score": None}
                log.append(f"CF fold-in from {anchor_ids} -> {len(cf['candidates'])}")

        # 2) Semantic (+ popularity re-rank iff prefer_popular)
        if plan["strategy"] in ("semantic", "hybrid"):
            sem = self.b.semantic_search(plan["semantic_query"], n=self.sem_retrieve_n)["results"]
            #                                                      ^^^^^^^^^^^^^^^^^^^  <-- CHANGE 2: was self.sem_n
            if plan["prefer_popular"]:
                sem = self.rr.rerank(sem, pop_weight=self.pop_weight, n=self.sem_retrieve_n)
                #                                                       ^^^^^^^^^^^^^^^^^^^  <-- CHANGE 2: was self.sem_n
                log.append(f"semantic '{plan['semantic_query']}' + pop re-rank -> {len(sem)}")
            else:
                log.append(f"semantic '{plan['semantic_query']}' (hidden-gems, no re-rank) -> {len(sem)}")
            for r in sem:
                sc = r.get("blended_score", r["score"])
                if r["movieId"] in pool:
                    pool[r["movieId"]]["source"] += "+semantic"
                else:
                    pool[r["movieId"]] = {"source": "semantic", "score": sc}

        # 3) Hard constraints (Tool D)
        cand_ids = list(pool.keys())
        exclude_ids = [mid for t in plan["exclude_titles"] if (mid := self.resolve_title(t))]
        yr = (plan["year_min"], plan["year_max"]) if (plan["year_min"] or plan["year_max"]) else None
        filt = self.d.filter_by(cand_ids, runtime_max=plan["runtime_max"], runtime_min=plan["runtime_min"],
                                genre_in=plan["genres_include"] or None, year_range=yr, exclude_ids=exclude_ids)
        log.append(f"filter {filt['n_in']} -> {filt['n_out']}")

        # 4) Cap + enrich with metadata (Tool C)
        enriched = []
        for mid in filt["filtered"][:self.cap]:
            det = self.c.get_movie_details(mid)
            if det.get("found"):
                det["source"] = pool[mid]["source"]
                det["retrieval_score"] = pool[mid]["score"]
                enriched.append(det)
        log.append(f"enriched {len(enriched)} candidates")
        return enriched, log


def retriever_node(state, retriever):
    cands, log = retriever.retrieve(state["plan"])
    return {"candidates": cands, "trajectory": ["RETRIEVER: " + " | ".join(log)]}
