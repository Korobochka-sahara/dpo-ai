from fastapi.testclient import TestClient

from src.api.api import app


client = TestClient(app)


def test_root():

    response = client.get("/")

    assert response.status_code == 200


def test_health():

    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_predict():

    response = client.post(
        "/predict",
        json={
            "text": "This product is great"
        }
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data

    assert "confidence" in data


def test_predict_empty_text():

    response = client.post(
        "/predict",
        json={
            "text": ""
        }
    )

    # если используешь min_length=1
    assert response.status_code == 422