"""Item-based collaborative filtering: latent-factor movie embeddings via TruncatedSVD
over the (movie x user) ratings matrix, then top-K nearest neighbors per movie by
cosine distance in latent space. Stored as a long table (movieId, neighbor_movieId,
similarity) so the app can look up a movie's neighbors without holding a dense
N x N similarity matrix in memory.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"

MIN_RATINGS_PER_MOVIE = 3   # drop movies with too few ratings to be reliable
N_COMPONENTS = 50           # latent factors for SVD
TOP_K_NEIGHBORS = 30


def build_collaborative_similarity() -> None:
    ratings = pd.read_parquet(ARTIFACTS_DIR / "ratings.parquet")

    counts = ratings.groupby("movieId").size()
    keep_movie_ids = counts[counts >= MIN_RATINGS_PER_MOVIE].index
    ratings = ratings[ratings["movieId"].isin(keep_movie_ids)]

    pivot = ratings.pivot_table(index="movieId", columns="userId", values="rating", fill_value=0)
    movie_ids = pivot.index.to_numpy()

    n_components = min(N_COMPONENTS, min(pivot.shape) - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    movie_factors = svd.fit_transform(pivot.values)

    n_neighbors = min(TOP_K_NEIGHBORS + 1, len(movie_ids))  # +1 because a movie is its own nearest neighbor
    nn = NearestNeighbors(n_neighbors=n_neighbors, metric="cosine")
    nn.fit(movie_factors)
    distances, indices = nn.kneighbors(movie_factors)

    rows = []
    for row_i, movie_id in enumerate(movie_ids):
        for dist, neighbor_row_i in zip(distances[row_i], indices[row_i]):
            neighbor_id = movie_ids[neighbor_row_i]
            if neighbor_id == movie_id:
                continue
            similarity = 1.0 - dist  # cosine distance -> similarity
            rows.append((movie_id, neighbor_id, similarity))

    neighbors_df = pd.DataFrame(rows, columns=["movieId", "neighbor_movieId", "similarity"])
    neighbors_df.to_parquet(ARTIFACTS_DIR / "collab_neighbors.parquet")
    print(f"collaborative neighbors: {len(movie_ids)} movies, {len(neighbors_df)} edges "
          f"-> {ARTIFACTS_DIR / 'collab_neighbors.parquet'}")


if __name__ == "__main__":
    build_collaborative_similarity()
