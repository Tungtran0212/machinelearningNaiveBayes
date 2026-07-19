import string

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


EXTRA_STOPWORDS = [
    "u",
    "ü",
    "ur",
    "4",
    "2",
    "im",
    "dont",
    "doin",
    "ure",
]


def text_preprocessing(message):
    if not isinstance(message, str):
        message = str(message)

    lemmatizer = WordNetLemmatizer()

    stop_words = set(
        stopwords.words("english") + EXTRA_STOPWORDS
    )

    # Remove punctuation
    message_without_punctuation = "".join(
        character
        for character in message
        if character not in string.punctuation
    )

    words = message_without_punctuation.split()

    cleaned_words = [
        lemmatizer.lemmatize(word.lower(), pos="v")
        for word in words
        if word.lower() not in stop_words
    ]

    return " ".join(cleaned_words)