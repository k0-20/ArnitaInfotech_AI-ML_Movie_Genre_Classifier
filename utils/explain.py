import joblib
import numpy as np
from utils.preprocessing import clean_text

model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def explain_prediction(movie_plot, top_n=10):
    cleaned = clean_text(movie_plot)
    vector = vectorizer.transform([cleaned])
    prediction = model.predict(vector)[0]

    feature_names = np.array(vectorizer.get_feature_names_out())
    values = vector.toarray()[0]

    non_zero = values.nonzero()[0]

    if len(non_zero) == 0:
        return prediction, []

    sorted_indices = non_zero[np.argsort(values[non_zero])[::-1]]
    top_words = feature_names[sorted_indices][:top_n]

    return prediction, list(top_words)