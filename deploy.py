from pathlib import Path

import joblib
import nltk
import streamlit as st

# This import must exist before joblib.load()
from multinomial_naive_bayes import CustomMultinomialNB
from preprocessing import text_preprocessing
from sklearn.feature_extraction.text import CountVectorizer

PROJECT_DIRECTORY = Path(__file__).resolve().parent

MODEL_FILE = (    PROJECT_DIRECTORY
    / "multinomial_nb_classifier.joblib"
)

VECTORIZER_FILE = (
    PROJECT_DIRECTORY
    / "count_vectorizer.joblib")


st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="centered"
)


if "update_message" in st.session_state:
    st.success(
        st.session_state.pop("update_message")
    )


def expand_vocabulary(vectorizer, cleaned_text):
    old_vocabulary = dict(vectorizer.vocabulary_)

    message_tokens = cleaned_text.split()

    new_tokens = sorted(
        {
            token
            for token in message_tokens
            if token not in old_vocabulary
        }
    )

    if not new_tokens:
        return vectorizer, []

    expanded_vocabulary = old_vocabulary.copy()
    next_feature_index = len(expanded_vocabulary)

    for token in new_tokens:
        expanded_vocabulary[token] = next_feature_index
        next_feature_index += 1

    expanded_vectorizer = CountVectorizer(
        vocabulary=expanded_vocabulary,
        lowercase=False,
    )

    expanded_vectorizer.transform([""])

    expected_size = len(old_vocabulary) + len(new_tokens)
    actual_size = len(expanded_vectorizer.vocabulary_)

    if actual_size != expected_size:
        raise RuntimeError(
            "Vocabulary expansion failed: "
            f"expected {expected_size}, got {actual_size}"
        )

    return expanded_vectorizer, new_tokens

@st.cache_resource
def load_artifacts():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

    if not VECTORIZER_FILE.exists():
        raise FileNotFoundError(
          f"Vectorizer file not found: {VECTORIZER_FILE}"
        )

    classifier = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    return classifier, vectorizer


def save_artifacts(classifier, vectorizer):
    joblib.dump(classifier, MODEL_FILE)
    joblib.dump(vectorizer, VECTORIZER_FILE)



def prepare_message(
    raw_text,
    vectorizer,
    allow_new_words=False,
):
    cleaned_text = text_preprocessing(raw_text)
    new_tokens = []

    if allow_new_words:
        vectorizer, new_tokens = expand_vocabulary(
            vectorizer,
            cleaned_text,
        )

    message_features = vectorizer.transform(
        [cleaned_text]
    )

    return (
        cleaned_text,
        message_features,
        vectorizer,
        new_tokens,
    )



try:
    classifier, vectorizer = load_artifacts()
except Exception as error:
    st.error("The model could not be loaded.")
    st.exception(error)
    st.stop()


st.title(" Email Spam Detector")

st.write(
    "This application uses a custom Multinomial Naive Bayes "
    "classifier coded from scratch."
)

with st.expander("Model information"):
    st.write(
        "Classifier:",
        type(classifier).__name__
    )
    st.write(
        "Vocabulary size:",
        len(vectorizer.vocabulary)
    )


email_text = st.text_area(
    "Enter an email message",
    height=220,
    placeholder=(
        "Example: Congratulations! You have won a free prize. "
       "Click here to claim it."
   )
)


if st.button("Classify email", type="primary"):
    if not email_text.strip():
        st.warning("Please enter an email message.")
    else:
        try:
            (
                cleaned_message,
                message_features,
                _,
                _,
            ) = prepare_message(
                email_text,
                vectorizer,
                allow_new_words=False,
            )

            prediction = classifier.predict(
                message_features
            )[0]

            label = "Spam" if prediction == 1 else "Ham"

            st.success(f"Prediction: {label}")

        except Exception as error:
            st.error("The email could not be classified.")
            st.exception(error)



st.subheader("Update model")
choice = st.radio(
    "Choose the correct label:",
    ("Spam", "Ham"),
)

y_new = 1 if choice == "Spam" else 0

if st.button("Add message and update model"):
    if not email_text.strip():
        st.warning("Please enter an email message.")
    else:
        try:
            old_size = len(vectorizer.vocabulary_)

            (
                cleaned_message,
                message_features,
                expanded_vectorizer,
                new_tokens,
            ) = prepare_message(
                email_text,
                vectorizer,
                allow_new_words=True,
            )

            new_size = len(
                expanded_vectorizer.vocabulary_
            )

            classifier.partial_fit_update(
                message_features,
                [y_new],
            )

            save_artifacts(
                classifier,
                expanded_vectorizer,
            )

            st.session_state["update_message"] = (
                f"Vocabulary expanded from "
                f"{old_size} to {new_size}. "
                f"Added words: {new_tokens}"
            )
                        
            st.write(
                "Vectorizer vocabulary:",
                len(expanded_vectorizer.vocabulary_),
            )

            st.write(
                "Message feature columns:",
                message_features.shape[1],
            )

            st.write(
                "Classifier feature columns:",
                classifier._feature_counts.shape[1],
            )

            load_artifacts.clear()
            st.rerun()

        except Exception as error:
            st.error("The model could not be updated.")
            st.exception(error)
