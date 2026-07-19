from pathlib import Path

import joblib
import nltk
import streamlit as st

# This import must exist before joblib.load()
from multinomial_naive_bayes import CustomMultinomialNB
from preprocessing import text_preprocessing

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


@st.cache_resource
def load_artifacts():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            f"Model file not found: {MODEL_FILE}"
        )

    if not VECTORIZER_FILE.exists():
       raise FileNotFoundError(
           f"Vectorizer file not found {VECTORIZER_FILE}"
        )

    classifier = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    return classifier, vectorizer


def prepare_message(raw_text, vectorizer):
    cleaned_text = text_preprocessing(raw_text)

    # Use transform(), not fit_transform()
    vectorized_message = vectorizer.transform(
        [cleaned_text]
    )

    return cleaned_text, vectorized_message


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
            cleaned_message, message_features = prepare_message(
                email_text,
                vectorizer
            )

            prediction = classifier.predict(message_features)[0]

            with st.expander("Preprocessed message"):
               st.code(cleaned_message)

            if prediction == 1 or str(prediction).lower() == "spam":
               st.error("Prediction: Spam")
            else:
               st.success("Prediction: Not Spam")

            st.caption(f"Raw predicted class: {prediction}")
        except LookupError:
           st.error(
                "An NLTK resource is missing. Run the NLTK "
                "download commands shown below."
            )
        except Exception as error:
            st.error("Prediction failed.")
            st.exception(error)

if st.button("Classify email", type="primary"):
    if not email_text.strip():
        st.warning("Please enter an email message.")
    else:
        try:
            cleaned_message, message_features = prepare_message(
                email_text,
                vectorizer
            )

            prediction = classifier.predict(message_features)[0]

            with st.expander("Preprocessed message"):
               st.code(cleaned_message)

            if prediction == 1 or str(prediction).lower() == "spam":
               st.error("Prediction: Spam")
            else:
               st.success("Prediction: Not Spam")

            st.caption(f"Raw predicted class: {prediction}")
        except LookupError:
           st.error(
                "An NLTK resource is missing. Run the NLTK "
                "download commands shown below."
            )
        except Exception as error:
            st.error("Prediction fiiled.")
            st.exception(error)