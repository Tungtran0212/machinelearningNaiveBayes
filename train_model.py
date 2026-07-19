from pathlib import Path

import joblib
import nltk
import pandas as pd

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from multinomial_naive_bayes import CustomMultinomialNB


PROJECT_DIRECTORY = Path(__file__).resolve().parent
DATA_FILE = PROJECT_DIRECTORY / "emails.csv"
MODEL_FILE = PROJECT_DIRECTORY / "multinomial_nb_classifier.joblib"
VECTORIZER_FILE = PROJECT_DIRECTORY / "count_vectorizer.joblib"


def download_nltk_resources():
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)


def main():
    download_nltk_resources()

    print(f"Reading dataset: {DATA_FILE}")

    df = pd.read_csv(DATA_FILE)

    print(f"Dataset shape: {df.shape}")
    print(f"Columns: {len(df.columns)}")

    if "Prediction" not in df.columns:
        raise ValueError(
            "The dataset must contain a column named 'Prediction'."
        )

    # Same feature selection used in your original notebook
    X = df.iloc[:, 1:-1]
    y = df["Prediction"]

    print(f"Number of samples: {len(X)}")
    print(f"Number of model features: {X.shape[1]}")
    print("Class distribution:")
    print(y.value_counts())

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=1,
        stratify=y
    )

    classifier = CustomMultinomialNB(alpha=1.0)

    classifier.fit(X_train, y_train)

    accuracy = classifier.score(X_test, y_test)
    predictions = classifier.predict(X_test)

    print(f"\nAccuracy: {accuracy:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion matrix:")
    print(confusion_matrix(y_test, predictions))

    # Use exactly the same vocabulary/order as the dataset columns
    model_vocabulary = list(X.columns)

    vectorizer = CountVectorizer(
        vocabulary=model_vocabulary
    )

    # Save the trained custom classifier
    joblib.dump(classifier, MODEL_FILE)

    # Save the vocabulary-compatible vectorizer
    joblib.dump(vectorizer, VECTORIZER_FILE)

    print(f"\nSaved classifier to: {MODEL_FILE}")
    print(f"Saved vectorizer to: {VECTORIZER_FILE}")

    # Test loading immediately
    loaded_classifier = joblib.load(MODEL_FILE)
    loaded_vectorizer = joblib.load(VECTORIZER_FILE)

    print(
        "Loaded classifier:",
        type(loaded_classifier).__name__
    )
    print(
        "Loaded vectorizer:",
        type(loaded_vectorizer).__name__
    )
    print("Model and vectorizer were saved successfully.")


if __name__ == "__main__":
    main()