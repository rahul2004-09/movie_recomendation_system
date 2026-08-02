"""Content-based similarity: bag-of-entities over each movie's "soup" (genres,
keywords, cast, director - see data_pipeline.make_soup), then cosine similarity
between all movies.

Uses CountVectorizer rather than TF-IDF deliberately: the soup is squashed, curated
tags/names (not natural-language prose), so IDF down-weighting of "common" tokens
isn't the right model here - a shared genre or director should count consistently
regardless of how common that genre is across the catalog.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def build_content_similarity() -> None:
    df = pd.read_parquet(ARTIFACTS_DIR / "movies_content.parquet")

    vectorizer = CountVectorizer(max_features=20000)
    count_matrix = vectorizer.fit_transform(df["soup"])

    sim_matrix = cosine_similarity(count_matrix, dense_output=True).astype(np.float32)

    np.save(ARTIFACTS_DIR / "content_sim.npy", sim_matrix)
    df[["id", "title"]].reset_index().rename(columns={"index": "content_idx"}).to_parquet(
        ARTIFACTS_DIR / "content_index.parquet"
    )
    print(f"content similarity matrix: {sim_matrix.shape} -> {ARTIFACTS_DIR / 'content_sim.npy'}")


if __name__ == "__main__":
    build_content_similarity()
