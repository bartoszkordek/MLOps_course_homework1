from typing import Any, Generator

import pytest
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from starlette.testclient import TestClient

import inference
from app import app


@pytest.fixture(scope="session")
def models() -> tuple[LogisticRegression, SentenceTransformer]:
    classifier = inference.load_classifier_model()
    transformer = inference.load_sentence_transformer_model()
    return classifier, transformer


@pytest.fixture(scope="session")
def client() -> Generator[TestClient, Any, None]:
    with TestClient(app) as client:
        yield client
