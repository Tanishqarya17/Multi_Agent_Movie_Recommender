"""Tool A — ALS collaborative-filtering candidate generator."""
import os
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, load_npz
from implicit.cpu.als import AlternatingLeastSquares  


class ToolA:
    def __init__(self, artifacts_dir, movies_csv, user_item_path=None):
        self.model = AlternatingLeastSquares.load(os.path.join(artifacts_dir, "als_model.npz"))
        maps = np.load(os.path.join(artifacts_dir, "als_mappings.npz"))
        self.item_categories = maps["item_categories"]
        self.user_categories = maps["user_categories"]
        self.n_items = len(self.item_categories)
        self.i_map = {int(m): j for j, m in enumerate(self.item_categories)}
        self.u_map = {int(u): i for i, u in enumerate(self.user_categories)}
        self._movies = pd.read_csv(movies_csv)[["movieId", "title", "genres"]].set_index("movieId")
        self.user_item = load_npz(user_item_path) if user_item_path else None

    def titles(self, movie_ids):
        present = [m for m in movie_ids if m in self._movies.index]
        return self._movies.loc[present].reset_index().to_dict("records")

    def candidates_from_anchors(self, anchor_movie_ids, n=200):
        idxs = [self.i_map[m] for m in anchor_movie_ids if m in self.i_map]
        dropped = [m for m in anchor_movie_ids if m not in self.i_map]
        if not idxs:
            return {"candidates": [], "dropped_anchors": dropped}
        row = csr_matrix((np.ones(len(idxs), "float32"),
                          (np.zeros(len(idxs), "int32"), np.array(idxs, "int32"))),
                         shape=(1, self.n_items))
        ids, _ = self.model.recommend(0, row, N=n,
                                      filter_already_liked_items=True, recalculate_user=True)
        return {"candidates": [int(self.item_categories[i]) for i in ids],
                "dropped_anchors": dropped}

    def candidates_for_user(self, user_id, n=200):
        if self.user_item is None:
            raise RuntimeError("demo-user mode needs user_item_path")
        if user_id not in self.u_map:
            return {"candidates": [], "error": "unknown user_id"}
        u = self.u_map[user_id]
        ids, _ = self.model.recommend(u, self.user_item[u], N=n, filter_already_liked_items=True)
        return {"candidates": [int(self.item_categories[i]) for i in ids]}

    def similar_movies(self, movie_id, n=20):
        if movie_id not in self.i_map:
            return {"similar": [], "error": "movie not in ALS space"}
        ids, _ = self.model.similar_items(self.i_map[movie_id], N=n + 1)
        out = [int(self.item_categories[j]) for j in ids if int(self.item_categories[j]) != movie_id]
        return {"similar": out[:n]}
