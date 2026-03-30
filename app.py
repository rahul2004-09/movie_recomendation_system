import streamlit as st  
import pickle 

movies = pickle.load(open("movie_list.pkl",'rb'))
similarity = pickle.load(open("similarity.pkl",'rb'))
movies_list = movies['title'].values


st.header("Movie Recomendation System")
st.selectbox("select movie from dropdown ", movies_list)

def recommand(movies):
    index = movies[movies['title']== movies].index[0]
    distance = sorted(list(enumerate(similarity[index])), reverse = True, key = lambda vector:vector[1])
    recommend_movie=[]
    for i in distance[0:5]:
        recommend_movie.append(movies.iloc[i[0]].title)
    return recommend_movie       


if st.button("Show recommend"):
    movie_name = recommend(selectvalue)
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.text(movie_name[0])
        


