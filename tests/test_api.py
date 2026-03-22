from fastapi import status
from fastapi.testclient import TestClient


def test_welcome_root(client: TestClient) -> None:
    response = client.get("/")
    assert response.status_code == status.HTTP_200_OK
    assert response.json() == {"message": "Welcome to the ML API"}


def test_health_status(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == status.HTTP_200_OK
    data = response.json()
    assert data["status"] == "ok"


def test_predict_endpoint_success(client: TestClient) -> None:
    response = client.post("/predict", json={"text": "I love MLOps!"})
    assert response.status_code == 200
    assert "prediction" in response.json()
    assert isinstance(response.json()["prediction"], str)


def test_predict_validation_error(client: TestClient) -> None:
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422
    assert "detail" in response.json()
