"""
Downloads and cleans the two Kaggle datasets this project needs:
  - TMDB 5000 Movie Dataset (content metadata: overview, genres, cast, crew, keywords)
  - MovieLens Small Latest Dataset (ratings.csv, for collaborative filtering)

Requires a Kaggle API token at ~/.kaggle/kaggle.json (or KAGGLE_USERNAME/KAGGLE_KEY
env vars). Run this once (or whenever you want to refresh data):

    python -m src.data_pipeline
"""

import ast
from pathlib import Path

import kagglehub
import pandas as pd

ARTIFACTS_DIR = Path(__file__).resolve().parent.parent / "artifacts"
ARTIFACTS_DIR.mkdir(exist_ok=True)


def _parse_list_field(raw: str, key: str = "name", limit: int | None = None) -> list[str]:
    """TMDB 5000 stores genres/keywords/cast/crew as stringified JSON lists of dicts."""
    try:
        items = ast.literal_eval(raw)
    except (ValueError, SyntaxError):
        return []
    names = [item[key] for item in items if key in item]
    return names[:limit] if limit else names


def _extract_director(raw_crew: str) -> str:
    try:
        crew = ast.literal_eval(raw_crew)
    except (ValueError, SyntaxError):
        return ""
    for member in crew:
        if member.get("job") == "Director":
            return member["name"]
    return ""


def build_content_dataset() -> pd.DataFrame:
    dataset_path = Path(kagglehub.dataset_download("tmdb/tmdb-movie-metadata"))
    movies = pd.read_csv(dataset_path / "tmdb_5000_movies.csv")
    credits = pd.read_csv(dataset_path / "tmdb_5000_credits.csv")

    credits = credits.rename(columns={"movie_id": "id"})[["id", "cast", "crew"]]
    df = movies.merge(credits, on="id")

    df["genres_list"] = df["genres"].apply(_parse_list_field)
    df["keywords_list"] = df["keywords"].apply(_parse_list_field)
    df["cast_list"] = df["cast"].apply(lambda x: _parse_list_field(x, limit=5))
    df["director"] = df["crew"].apply(_extract_director)
    df["overview"] = df["overview"].fillna("")

    def make_soup(row) -> str:
        """Metadata-only soup (no overview prose): each entity is squashed to a single
        token (e.g. "Science Fiction" -> "sciencefiction", "Tom Hanks" -> "tomhanks")
        so TF-IDF/CountVectorizer treats it as one unit instead of splitting it into
        common words that create false matches across unrelated movies/people.
        Genres and director are repeated for extra weight, since genre/director
        consistency matters more for "similar movie" recommendations than raw plot
        text does (plot-word overlap alone tends to produce noisy, off-genre matches).
        """
        def squash(s: str) -> str:
            return str(s).replace(" ", "").lower()

        parts = (
            [squash(g) for g in row["genres_list"]] * 2
            + [squash(k) for k in row["keywords_list"]]
            + [squash(c) for c in row["cast_list"]]
            + [squash(row["director"])] * 3
        )
        return " ".join(p for p in parts if p)

    df["soup"] = df.apply(make_soup, axis=1)

    keep_cols = [
        "id", "title", "overview", "release_date", "vote_average", "vote_count",
        "genres_list", "keywords_list", "cast_list", "director", "soup",
    ]
    df = df[keep_cols].dropna(subset=["title"]).reset_index(drop=True)
    df.to_parquet(ARTIFACTS_DIR / "movies_content.parquet")
    print(f"content dataset: {len(df)} movies -> {ARTIFACTS_DIR / 'movies_content.parquet'}")
    return df


def build_ratings_dataset() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    dataset_path = Path(kagglehub.dataset_download("shubhammehta21/movie-lens-small-latest-dataset"))
    ratings = pd.read_csv(dataset_path / "ratings.csv")
    ml_movies = pd.read_csv(dataset_path / "movies.csv")
    links = pd.read_csv(dataset_path / "links.csv")

    ratings.to_parquet(ARTIFACTS_DIR / "ratings.parquet")
    ml_movies.to_parquet(ARTIFACTS_DIR / "ml_movies.parquet")
    links.to_parquet(ARTIFACTS_DIR / "links.parquet")
    print(f"ratings dataset: {len(ratings)} ratings, {len(ml_movies)} movies -> {ARTIFACTS_DIR}")
    return ratings, ml_movies, links


def bridge_datasets(content_df: pd.DataFrame, links: pd.DataFrame) -> pd.DataFrame:
    """Maps MovieLens movieId <-> TMDB id via links.csv's tmdbId column."""
    links = links.dropna(subset=["tmdbId"]).copy()
    links["tmdbId"] = links["tmdbId"].astype(int)
    bridge = links.merge(content_df[["id"]], left_on="tmdbId", right_on="id", how="inner")
    bridge = bridge[["movieId", "tmdbId"]].rename(columns={"tmdbId": "id"})
    bridge.to_parquet(ARTIFACTS_DIR / "bridge.parquet")
    print(f"bridge: {len(bridge)} movies matched between MovieLens and TMDB 5000")
    return bridge


def main() -> None:
    content_df = build_content_dataset()
    _, _, links = build_ratings_dataset()
    bridge_datasets(content_df, links)
    print("\nData pipeline done. Next: python -m src.build_models")


if __name__ == "__main__":
    main()
