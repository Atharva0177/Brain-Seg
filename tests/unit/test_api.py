from fastapi.testclient import TestClient

from app.api.main import PredictionRequest, app


def test_health_endpoint() -> None:
    response = TestClient(app).get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_prediction_request_requires_four_channels() -> None:
    request = PredictionRequest(images=[[[[0.0]]]] * 4)
    assert len(request.images) == 4
