"""Tool B — semantic plot search (Qwen3 embeddings + FAISS)."""
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

QUERY_PROMPT = ("Instruct: Given a movie search query describing plot, mood, or themes, "
                "retrieve movies whose plot matches.\nQuery:")


class ToolB:
    def __init__(self, faiss_dir, catalog_path, model_name="Qwen/Qwen3-Embedding-0.6B",
                 emb_dim=512, device=None):
        self.index = faiss.read_index(os.path.join(faiss_dir, "plots.index"))
        self.movieids = np.load(os.path.join(faiss_dir, "faiss_movieids.npy"))
        self.model = SentenceTransformer(model_name, truncate_dim=emb_dim,
                                         processor_kwargs={"padding_side": "left"},
                                         device=device)
        self._title = pd.read_parquet(catalog_path)[["movieId", "title"]].set_index("movieId")["title"]

    def semantic_search(self, query_text, n=50):
        q = self.model.encode([query_text], prompt=QUERY_PROMPT,
                              normalize_embeddings=True, convert_to_numpy=True).astype("float32")
        scores, idx = self.index.search(q, n)
        results = [{"movieId": int(self.movieids[j]), "score": float(s),
                    "title": self._title.get(int(self.movieids[j]), None)}
                   for j, s in zip(idx[0], scores[0])]
        return {"results": results}
