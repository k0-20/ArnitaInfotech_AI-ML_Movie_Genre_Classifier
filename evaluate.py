import os
import joblib
import pandas as pd

from utils.preprocessing import clean_text

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    classification_report
)

# ==========================
# LOAD DATASET
# ==========================

print("=" * 60)
print("MOVIE GENRE CLASSIFIER - MODEL COMPARISON")
print("=" * 60)

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

print("\nDataset Loaded Successfully")
print("Total Movies:", len(df))

# ==========================
# CLEAN TEXT
# ==========================

print("\nCleaning Text...")

df["clean_description"] = df["DESCRIPTION"].apply(clean_text)

# ==========================
# TF-IDF
# ==========================

print("\nCreating TF-IDF Features...")

vectorizer = TfidfVectorizer(
    max_features=5000
)

X = vectorizer.fit_transform(df["clean_description"])

y = df["GENRE"]

# ==========================
# SPLIT
# ==========================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)

# ==========================
# MODELS
# ==========================

models = {

    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "Linear SVM": LinearSVC()

}

results = []

best_accuracy = 0
best_model = None
best_name = ""

os.makedirs(
    "models",
    exist_ok=True
)

print("\nTraining Models...\n")

# ==========================
# TRAIN ALL MODELS
# ==========================

for name, model in models.items():

    print(f"Training {name}...")

    model.fit(
        X_train,
        y_train
    )

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "Accuracy": accuracy
    })

    print(f"{name} Accuracy : {accuracy:.4f}")

    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    file_name = name.lower().replace(" ", "_") + ".pkl"

    joblib.dump(
        model,
        f"models/{file_name}"
    )

    if accuracy > best_accuracy:

        best_accuracy = accuracy
        best_model = model
        best_name = name

# ==========================
# SAVE BEST MODEL
# ==========================

joblib.dump(
    best_model,
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

# ==========================
# SAVE RESULTS
# ==========================

results_df = pd.DataFrame(results)

results_df.to_csv(
    "models/model_results.csv",
    index=False
)

# ==========================
# DISPLAY RESULTS
# ==========================

print("\n" + "=" * 60)

print("MODEL COMPARISON")

print("=" * 60)

print(results_df)

print("\nBest Model :", best_name)

print("Best Accuracy :", round(best_accuracy * 100, 2), "%")

print("\nEverything Saved Successfully!")

print("=" * 60)