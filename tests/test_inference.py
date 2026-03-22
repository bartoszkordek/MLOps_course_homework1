import pytest
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

from inference import transform, predict


def test_models_loading(models: tuple[LogisticRegression, SentenceTransformer]) -> None:
    classifier, transformer = models
    assert classifier is not None
    assert transformer is not None


@pytest.mark.parametrize(
    "text",
    [
        "What a great MLOps lecture, I am very satisfied",
        "I feel bad",
        "I need to go out",
        "I feel so happy",
    ],
)
def test_inference_logic(
    models: tuple[LogisticRegression, SentenceTransformer], text: str
) -> None:
    classifier, transformer = models

    embedding = transform(transformer, text)
    result = predict(classifier, embedding)

    assert result in ["positive", "neutral", "negative"]
