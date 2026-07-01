import streamlit as st

from utils.predictor import predict_genre
from utils.recommender import recommend_movies
from utils.explain import explain_prediction

st.set_page_config(
    page_title="Movie Genre Classifier",
    page_icon="🎬",
    layout="wide"
)

st.title("🎬 Movie Genre Classification using NLP")

st.write(
    "Predict the genre of a movie from its plot summary "
    "and receive similar movie recommendations."
)

plot = st.text_area(
    "Enter Movie Plot",
    height=220
)

if st.button("Predict Genre"):

    if plot.strip() == "":

        st.warning("Please enter a movie description.")

    else:

        genre, confidence = predict_genre(plot)

        st.success(f"🎯 Predicted Genre : {genre}")

        if confidence is not None:

            st.info(f"Confidence : {confidence}")

        st.divider()

        st.subheader("💡 Important Keywords")

        _, keywords = explain_prediction(plot)

        if keywords:

            cols = st.columns(5)

            for i, word in enumerate(keywords):

                cols[i % 5].metric("", word)

        st.divider()

        st.subheader("🎥 Similar Movies")

        recommendations = recommend_movies(plot)

        for movie in recommendations:

            st.write(
                f"**{movie['Title']}** "
                f"({movie['Genre']})"
                f" — Similarity: {movie['Similarity']}%"
            )