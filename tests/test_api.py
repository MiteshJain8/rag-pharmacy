from fastapi.testclient import TestClient

from app.api.v1.routes import get_rag_service
from app.main import app
from app.services.rag.pipeline import RetrievalResult


class FakeRagService:
    async def query(self, query: str, top_k: int, expand_query: bool) -> RetrievalResult:
        return RetrievalResult(
            answer=f"Answer for {query}",
            sources=[],
            retrieval={"expanded_queries": [query]},
        )


app.dependency_overrides[get_rag_service] = lambda: FakeRagService()

client = TestClient(app)


def test_root() -> None:
    response = client.get("/")
    assert response.status_code == 200
    assert "Indian Medicine Source Lookup" in response.text
    assert "% confidence" not in response.text
    assert "/api/v1/query" in response.text


def test_health() -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_query_contract() -> None:
    response = client.post("/api/v1/query", json={"query": "paracetamol substitute"})
    assert response.status_code == 200
    assert response.json()["retrieval"]["expanded_queries"] == ["paracetamol substitute"]
    assert "confidence" not in response.json()


def test_query_rejects_blank_text() -> None:
    response = client.post("/api/v1/query", json={"query": "   "})
    assert response.status_code == 422
