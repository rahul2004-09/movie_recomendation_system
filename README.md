# Hybrid Movie Recommender

Content-based (bag-of-entities over genres/keywords/cast/director + cosine similarity)
and collaborative filtering (SVD item embeddings + KNN over MovieLens ratings), blended
with a user-adjustable weight, served via Streamlit. Posters via TMDB, falling back to
Wikipedia automatically when TMDB is unconfigured or unreachable.

## Setup

### 1. Install dependencies
```
python -m pip install -r requirements.txt
```

### 2. Kaggle API credentials (needed to auto-download datasets)
1. Create a free account at https://www.kaggle.com
2. Go to https://www.kaggle.com/settings -> "API" -> "Create New Token" -> downloads `kaggle.json`
3. Place it at `C:\Users\<you>\.kaggle\kaggle.json`

### 3. TMDB API key (optional - posters fall back to Wikipedia without it)
1. Create a free account at https://www.themoviedb.org
2. Go to Settings -> API -> request a free API key (v3 auth)
3. Copy `.env.example` to `.env` and paste your key:
   ```
   copy .env.example .env
   ```
   then edit `.env` and set `TMDB_API_KEY=your_key_here`

### 4. Build the data + models (run once)
```
python -m src.data_pipeline
python -m src.build_models
```
This downloads the TMDB 5000 Movie Dataset and the MovieLens Small Latest Dataset
from Kaggle, then precomputes the content and collaborative similarity artifacts
into `artifacts/`. Takes a few minutes.

### 5. Run the app
```
streamlit run app.py
```

## How it works

- **Content-based**: each movie's genres + keywords + top cast + director are
  squashed into single tokens (e.g. "Science Fiction" -> "sciencefiction",
  "Tom Hanks" -> "tomhanks") so shared entities match as a whole instead of being
  split into individual common words, then vectorized with CountVectorizer and
  compared via cosine similarity. Genres and director are repeated for extra
  weight, since genre/director consistency matters more for "similar movie"
  recommendations than raw plot text does.
- **Collaborative filtering**: the MovieLens ratings matrix (movies x users) is
  reduced to 50 latent factors with TruncatedSVD, then each movie's nearest
  neighbors are found via cosine distance in that latent space (KNN).
- **Bridging**: MovieLens `links.csv` maps MovieLens `movieId` to TMDB `id`, so
  results from both models can be combined for the same movie.
- **Hybrid**: `final_score = alpha * content_score + (1 - alpha) * collab_score`,
  where `alpha` is a slider in the app. If a movie has no MovieLens rating data,
  the app automatically falls back to content-only and says so.
- **Posters**: tries TMDB first if `TMDB_API_KEY` is set; falls back to Wikipedia's
  public REST API (no key needed) if TMDB is unconfigured or unreachable.

## Project structure
```
src/
  data_pipeline.py   # downloads + cleans both Kaggle datasets
  content_based.py   # bag-of-entities + cosine similarity
  collaborative.py   # SVD + KNN item similarity
  hybrid.py           # combines both into ranked recommendations
  tmdb_api.py         # poster lookups (TMDB, falls back to Wikipedia)
  build_models.py     # orchestrates content_based + collaborative
app.py                # Streamlit UI
artifacts/             # generated data/model files (gitignored)
```
