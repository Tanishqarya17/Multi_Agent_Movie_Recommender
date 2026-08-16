"""Tool C — movie metadata lookup (offline, from the catalog)."""
import pandas as pd


class ToolC:
    def __init__(self, catalog_path, movies_csv):
        self.cat = pd.read_parquet(catalog_path).set_index("movieId")
        self.ml_genres = pd.read_csv(movies_csv).set_index("movieId")["genres"]  # MovieLens genres (complete)

    def _genres(self, movie_id):
        g = self.ml_genres.get(movie_id)
        return [] if (not isinstance(g, str) or g == "(no genres listed)") else g.split("|")

    def get_movie_details(self, movie_id):
        movie_id = int(movie_id)
        if movie_id not in self.cat.index:
            return {"movieId": movie_id, "found": False}
        r = self.cat.loc[movie_id]
        return {
            "movieId": movie_id, "found": True,
            "title":    r["title"],
            "year":     int(r["year"])    if pd.notna(r["year"])    else None,
            "runtime":  int(r["runtime"]) if pd.notna(r["runtime"]) else None,
            "genres":   self._genres(movie_id),
            "director": r["director"],
            "cast":     list(r["cast"]) if r["cast"] is not None else [],
            "overview": r["overview"],
        }

    def get_many(self, movie_ids):
        return [self.get_movie_details(m) for m in movie_ids]
