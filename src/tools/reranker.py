"""Agent-invokable popularity re-ranker. Blends a popularity prior into semantic candidates.
Kept separate from Tool B so the PLANNER decides when to apply it (popular vs hidden-gems)."""
import numpy as np
import pandas as pd


class PopularityReranker:
    def __init__(self, catalog_path):
        pop = pd.read_parquet(catalog_path).set_index("movieId")["rating_count"]
        self.pop = pop.to_dict()
        self._max_log = float(np.log1p(max(self.pop.values()))) if self.pop else 1.0

    def _prior(self, movie_id):                       # 0..1 popularity, log-scaled
        return float(np.log1p(self.pop.get(int(movie_id), 0)) / self._max_log)

    def rerank(self, results, pop_weight=0.15, n=None):
        ranked = []
        for r in results:                              # results = Tool B output (movieId + cosine 'score')
            prior = self._prior(r["movieId"])
            ranked.append({**r,
                           "popularity_prior": round(prior, 3),
                           "blended_score": round(r["score"] + pop_weight * prior, 4)})
        ranked.sort(key=lambda x: x["blended_score"], reverse=True)
        return ranked[:n] if n else ranked
