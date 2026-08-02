"""Fetches poster images. Tries TMDB first (needs TMDB_API_KEY in .env), falls back
to Wikipedia's free public REST API (no key or signup needed) when TMDB is
unconfigured or unreachable, and finally a placeholder image."""

import os
import urllib.parse

import requests
from dotenv import load_dotenv

load_dotenv()

TMDB_API_KEY = os.environ.get("TMDB_API_KEY")
TMDB_IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w342"
PLACEHOLDER_POSTER = "https://via.placeholder.com/342x513?text=No+Poster"


def _from_tmdb(tmdb_id: int) -> str | None:
    if not TMDB_API_KEY:
        return None
    try:
        resp = requests.get(
            f"https://api.themoviedb.org/3/movie/{tmdb_id}",
            params={"api_key": TMDB_API_KEY},
            timeout=5,
        )
        resp.raise_for_status()
        poster_path = resp.json().get("poster_path")
        return f"{TMDB_IMAGE_BASE_URL}{poster_path}" if poster_path else None
    except requests.RequestException:
        return None


def _from_wikipedia(title: str) -> str | None:
    """Tries "<title> (film)" first to avoid landing on an unrelated Wikipedia page
    that happens to share the plain title, then falls back to the bare title."""
    for candidate in (f"{title} (film)", title):
        try:
            resp = requests.get(
                "https://en.wikipedia.org/api/rest_v1/page/summary/"
                + urllib.parse.quote(candidate, safe=""),
                timeout=5,
                headers={"User-Agent": "movie-recommender-mini-project/1.0"},
            )
            if resp.status_code != 200:
                continue
            thumbnail = resp.json().get("thumbnail", {}).get("source")
            if thumbnail:
                return thumbnail
        except requests.RequestException:
            continue
    return None


def get_poster_url(tmdb_id: int, title: str) -> str:
    return _from_tmdb(tmdb_id) or _from_wikipedia(title) or PLACEHOLDER_POSTER
