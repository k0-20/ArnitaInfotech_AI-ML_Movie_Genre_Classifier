import os
import joblib
import pandas as pd

from utils.preprocessing import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("=" * 50)
print("MOVIE GENRE CLASSIFIER")
print("=" * 50)

# -----------------------
# LOAD DATASET
# -----------------------

print("\nLoading Dataset...")

df = pd.read_csv(
    "data/train_data.txt",
    sep=" ::: ",
    names=[
        "ID",
        "TITLE",
        "GENRE",
        "DESCRIPTION"
    ],
    engine="python"
)

print("Dataset Loaded Successfully")

print("Total Movies:", len(df))

# -----------------------
# CLEAN TEXT
# -----------------------

print("\nCleaning Movie Descriptions...")

df["clean_description"] = df["DESCRIPTION"].apply(clean_text)

# -----------------------
# TF-IDF
# -----------------------

print("\nCreating TF-IDF Features...")

vectorizer = TfidfVectorizer(
    max_features=5000
)

X = vectorizer.fit_transform(
    df["clean_description"]
)

y = df["GENRE"]

print("Feature Matrix Shape:", X.shape)

# -----------------------
# TRAIN TEST SPLIT
# -----------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

print("\nTraining Samples:", X_train.shape[0])
print("Testing Samples :", X_test.shape[0])

# -----------------------
# TRAIN MODEL
# -----------------------

print("\nTraining Naive Bayes Model...")

model = MultinomialNB()

model.fit(
    X_train,
    y_train
)

print("Training Complete!")

# -----------------------
# PREDICT
# -----------------------

predictions = model.predict(
    X_test
)

# -----------------------
# EVALUATION
# -----------------------

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\nAccuracy")

print(round(accuracy * 100, 2), "%")

print("\nClassification Report")

print(
    classification_report(
        y_test,
        predictions
    )
)

print("\nConfusion Matrix")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# -----------------------
# SAVE MODEL
# -----------------------

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    model,
    "models/best_model.pkl"
)

joblib.dump(
    vectorizer,
    "models/vectorizer.pkl"
)

joblib.dump(
    df,
    "models/movie_database.pkl"
)

print("\nEverything Saved Successfully!")