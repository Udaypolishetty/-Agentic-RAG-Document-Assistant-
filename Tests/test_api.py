from fastapi.testclient import TestClient

from app.api.main import app


client = TestClient(app)


def test_home():

    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["message"] == "Agentic RAG API is running"


def test_empty_question():

    response = client.post(
        "/ask",
        json={"question": ""}
    )

    assert response.status_code == 400