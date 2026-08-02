from pathlib import Path

import streamlit as st

from src.hybrid import HybridRecommender
from src.tmdb_api import get_poster_url

ARTIFACTS_DIR = Path(__file__).resolve().parent / "artifacts"
REQUIRED_ARTIFACTS = [
    "movies_content.parquet", "content_index.parquet", "content_sim.npy",
    "bridge.parquet", "collab_neighbors.parquet",
]

st.set_page_config(page_title="Movie Recommender", page_icon="🎬", layout="wide")


@st.cache_resource
def load_recommender() -> HybridRecommender:
    return HybridRecommender()


@st.cache_data(show_spinner=False)
def cached_poster_url(tmdb_id: int, title: str) -> str:
    return get_poster_url(tmdb_id, title)


def artifacts_missing() -> list[str]:
    return [f for f in REQUIRED_ARTIFACTS if not (ARTIFACTS_DIR / f).exists()]


def main() -> None:
    st.title("🎬 Hybrid Movie Recommender")
    st.caption("Content-based (bag-of-entities) + Collaborative filtering (SVD), blended.")

    missing = artifacts_missing()
    if missing:
        st.error(
            "Missing precomputed artifacts: " + ", ".join(missing) +
            "\n\nRun these first, from the project root:\n\n"
            "```\npython -m src.data_pipeline\npython -m src.build_models\n```"
        )
        st.stop()

    recommender = load_recommender()

    col_left, col_right = st.columns([2, 1])
    with col_left:
        title = st.selectbox("Pick a movie you like", options=recommender.titles())
    with col_right:
        top_n = st.slider("How many recommendations", min_value=5, max_value=20, value=10)

    alpha = st.slider(
        "Content-based ↔ Collaborative-filtering blend",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="1.0 = pure content similarity (genres/plot/cast). 0.0 = pure collaborative "
             "(what similar users also liked). 0.5 = even blend.",
    )

    if not recommender.has_rating_data(title):
        st.info(f"No MovieLens rating data for **{title}** — showing content-based results only.")

    if st.button("Recommend", type="primary"):
        results, used_hybrid = recommender.recommend(title, alpha=alpha, top_n=top_n)

        if not results:
            st.warning("No recommendations found for that title.")
            return

        if used_hybrid:
            st.success(f"Blended recommendations (alpha={alpha:.2f}) for **{title}**")
        else:
            st.success(f"Content-based recommendations for **{title}**")

        cols = st.columns(5)
        for i, rec in enumerate(results):
            with cols[i % 5]:
                st.image(cached_poster_url(rec.tmdb_id, rec.title), use_container_width=True)
                st.markdown(f"**{rec.title}**")
                st.caption(f"score {rec.score:.2f}  ·  content {rec.content_score:.2f}  ·  collab {rec.collab_score:.2f}")


if __name__ == "__main__":
    main()
