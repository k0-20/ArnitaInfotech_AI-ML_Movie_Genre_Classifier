import joblib

from sklearn.metrics.pairwise import cosine_similarity

from utils.preprocessing import clean_text

# ==========================
# LOAD SAVED FILES
# ==========================

vectorizer = joblib.load("models/vectorizer.pkl")

movie_database = joblib.load("models/movie_database.pkl")

# Create TF-IDF vectors for all movies
movie_vectors = vectorizer.transform(
    movie_database["clean_description"]
)


# ==========================
# RECOMMEND MOVIES
# ==========================

def recommend_movies(user_plot, top_n=5):

    cleaned_plot = clean_text(user_plot)

    user_vector = vectorizer.transform([cleaned_plot])

    similarity = cosine_similarity(
        user_vector,
        movie_vectors
    )

    indices = similarity.argsort()[0][-top_n:][::-1]

    recommendations = []

    for index in indices:

        recommendations.append({

            "Title": movie_database.iloc[index]["TITLE"],

            "Genre": movie_database.iloc[index]["GENRE"],

            "Similarity": round(
                similarity[0][index] * 100,
                2
            )

        })

    return recommendations


# ==========================
# TEST
# ==========================

if __name__ == "__main__":

    plot = input("Enter Movie Plot:\n\n")

    movies = recommend_movies(plot)

    print("\nTop Recommended Movies\n")

    for i, movie in enumerate(movies, start=1):

        print(
            f"{i}. {movie['Title']} "
            f"({movie['Genre']}) "
            f"- {movie['Similarity']}%"
        )