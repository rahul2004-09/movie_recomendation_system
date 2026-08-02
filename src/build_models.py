"""Runs the full offline pipeline: fetch data, then build both similarity artifacts.
Run this once after data_pipeline.py (or use `python -m src.build_models` to do both).

    python -m src.data_pipeline
    python -m src.build_models
"""

from src.collaborative import build_collaborative_similarity
from src.content_based import build_content_similarity


def main() -> None:
    build_content_similarity()
    build_collaborative_similarity()
    print("\nAll artifacts built. Run: streamlit run app.py")


if __name__ == "__main__":
    main()
