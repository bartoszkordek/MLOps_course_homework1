from typing import Any

import joblib
from numpy import ndarray
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

SENTIMENT_MAP = {0: "negative", 1: "neutral", 2: "positive"}


def load_classifier_model() -> LogisticRegression:
    classifier = joblib.load("models/classifier.joblib")
    if not isinstance(classifier, LogisticRegression):
        raise TypeError("Loaded object is not a LogisticRegression")
    return classifier


def load_sentence_transformer_model() -> SentenceTransformer:
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    if not isinstance(encoder, SentenceTransformer):
        raise TypeError("Loaded object is not a SentenceTransformer")
    return encoder


def transform(model: SentenceTransformer, text: str) -> ndarray:
    sentences = [text]
    return model.encode(sentences)


def predict(model: LogisticRegression, embedding: ndarray) -> Any:
    prediction_id = int(model.predict(embedding.reshape(1, -1))[0])
    return SENTIMENT_MAP[prediction_id]
