"""Combines content-based similarity and collaborative-filtering similarity into a
single ranked recommendation list, weighted by `alpha` (1.0 = pure content-based,
0.0 = pure collaborative).

If the selected movie has no MovieLens rating data (no bridge entry), collaborative
scores aren't available for it, so we fall back to content-only and say so.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

CANDIDATE_POOL_SIZE = 100  # how many top content matches to consider before hybrid re-ranking


@dataclass
class Recommendation:
    tmdb_id: int
    title: str
    score: float
    content_score: float
    collab_score: float


class HybridRecommender:
    def __init__(self) -> None:
        self.content_df = pd.read_parquet(ARTIFACTS_DIR / "movies_content.parquet")
        self.content_index = pd.read_parquet(ARTIFACTS_DIR / "content_index.parquet")
        self.content_sim = np.load(ARTIFACTS_DIR / "content_sim.npy")
        self.bridge = pd.read_parquet(ARTIFACTS_DIR / "bridge.parquet")  # movieId <-> id (tmdb)
        self.collab_neighbors = pd.read_parquet(ARTIFACTS_DIR / "collab_neighbors.parquet")

        self._title_to_idx = dict(zip(self.content_index["title"], self.content_index["content_idx"]))
        self._idx_to_id = dict(zip(self.content_index["content_idx"], self.content_index["id"]))
        self._id_to_movieid = dict(zip(self.bridge["id"], self.bridge["movieId"]))
        self._movieid_to_id = dict(zip(self.bridge["movieId"], self.bridge["id"]))

    def titles(self) -> list[str]:
        return sorted(self._title_to_idx.keys())

    def has_rating_data(self, title: str) -> bool:
        idx = self._title_to_idx.get(title)
        if idx is None:
            return False
        tmdb_id = self._idx_to_id[idx]
        return tmdb_id in self._id_to_movieid

    def recommend(self, title: str, alpha: float = 0.5, top_n: int = 10) -> tuple[list[Recommendation], bool]:
        """Returns (recommendations, used_hybrid). used_hybrid is False when the
        selected movie has no rating data and we silently fell back to content-only."""
        idx = self._title_to_idx.get(title)
        if idx is None:
            return [], False

        tmdb_id = self._idx_to_id[idx]
        content_scores = self.content_sim[idx]  # shape (N,), aligned with content_index rows

        top_idxs = np.argsort(-content_scores)[: CANDIDATE_POOL_SIZE + 1]
        top_idxs = [i for i in top_idxs if i != idx][:CANDIDATE_POOL_SIZE]

        movie_id = self._id_to_movieid.get(tmdb_id)
        collab_lookup: dict[int, float] = {}
        used_hybrid = False
        if movie_id is not None:
            neighbors = self.collab_neighbors[self.collab_neighbors["movieId"] == movie_id]
            if not neighbors.empty:
                used_hybrid = True
                for _, row in neighbors.iterrows():
                    neighbor_tmdb_id = self._movieid_to_id.get(row["neighbor_movieId"])
                    if neighbor_tmdb_id is not None:
                        collab_lookup[neighbor_tmdb_id] = float(row["similarity"])

        effective_alpha = alpha if used_hybrid else 1.0

        results = []
        for i in top_idxs:
            cand_tmdb_id = self._idx_to_id[i]
            c_score = float(content_scores[i])
            col_score = collab_lookup.get(cand_tmdb_id, 0.0)
            final = effective_alpha * c_score + (1 - effective_alpha) * col_score
            title_i = self.content_index.loc[self.content_index["content_idx"] == i, "title"].iloc[0]
            results.append(Recommendation(cand_tmdb_id, title_i, final, c_score, col_score))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_n], used_hybrid
