"""Tool D — constrained filter over a candidate movieId list."""
import pandas as pd


class ToolD:
    def __init__(self, catalog_path, movies_csv):
        cat = pd.read_parquet(catalog_path).set_index("movieId")
        self.runtime = cat["runtime"].to_dict()
        self.year    = cat["year"].to_dict()
        ml = pd.read_csv(movies_csv).set_index("movieId")["genres"]
        self.genres = {int(m): set(g.split("|")) for m, g in ml.items()
                       if isinstance(g, str) and g != "(no genres listed)"}

    def filter_by(self, movie_ids, runtime_max=None, runtime_min=None,
                  genre_in=None, year_range=None, exclude_ids=None):
        exclude  = {int(x) for x in (exclude_ids or [])}
        genre_in = set(genre_in) if genre_in else None
        ylo, yhi = year_range or (None, None)
        out = []
        for m in movie_ids:
            m = int(m)
            if m in exclude:
                continue
            rt = self.runtime.get(m); yr = self.year.get(m)
            if runtime_max is not None and (rt is None or pd.isna(rt) or rt > runtime_max): continue
            if runtime_min is not None and (rt is None or pd.isna(rt) or rt < runtime_min): continue
            if year_range is not None:
                if yr is None or pd.isna(yr): continue
                if ylo is not None and yr < ylo: continue
                if yhi is not None and yr > yhi: continue
            if genre_in is not None and not (self.genres.get(m, set()) & genre_in): continue
            out.append(m)
        return {"filtered": out, "n_in": len(movie_ids), "n_out": len(out)}
