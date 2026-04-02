import streamlit as st
import pickle
import requests

# ---------------- FETCH POSTER ---------------- #
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key=c7ec19ffdd3279641fb606d19ceb9bb1&language=en-US"
        data = requests.get(url).json()

        poster_path = data.get('poster_path')

        if poster_path:
            return "https://image.tmdb.org/t/p/w500/" + poster_path
        else:
            return "https://via.placeholder.com/500x750?text=No+Image"
    except:
        return "https://via.placeholder.com/500x750?text=Error"


# ---------------- LOAD DATA ---------------- #
movies = pickle.load(open("movie_list.pkl",'rb'))
similarity = pickle.load(open("similarity.pkl",'rb'))

movies_list = movies['title'].values

st.header("🎬 Movie Recommendation System")

# ---------------- SELECT MOVIE ---------------- #
selected_movie = st.selectbox("Select movie from dropdown", movies_list)


# ---------------- RECOMMEND FUNCTION ---------------- #
def recommend(movie):
    index = movies[movies['title'] == movie].index[0]

    distance = sorted(
        list(enumerate(similarity[index])),
        reverse=True,
        key=lambda vector: vector[1]
    )

    recommend_movie = []
    recommend_poster = []

    for i in distance[1:6]:  # skip itself
        movie_id = movies.iloc[i[0]].id   # make sure column is 'id'
        
        recommend_movie.append(movies.iloc[i[0]].title)
        recommend_poster.append(fetch_poster(movie_id))

    return recommend_movie, recommend_poster


# ---------------- BUTTON ---------------- #
if st.button("Show Recommend"):
    movie_names, movie_posters = recommend(selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.text(movie_names[0])
        st.image(movie_posters[0])

    with col2:
        st.text(movie_names[1])
        st.image(movie_posters[1])

    with col3:
        st.text(movie_names[2])
        st.image(movie_posters[2])

    with col4:
        st.text(movie_names[3])
        st.image(movie_posters[3])

    with col5:
        st.text(movie_names[4])
        st.image(movie_posters[4])