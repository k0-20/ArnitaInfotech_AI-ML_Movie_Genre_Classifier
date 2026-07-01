import joblib
from utils.preprocessing import clean_text

model = joblib.load("models/best_model.pkl")
vectorizer = joblib.load("models/vectorizer.pkl")

def predict_genre(movie_description):

    cleaned = clean_text(movie_description)

    vector = vectorizer.transform([cleaned])

    prediction = model.predict(vector)[0]

    confidence = None

    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(vector)[0]
        confidence = round(max(probabilities) * 100, 2)

    return prediction, confidence