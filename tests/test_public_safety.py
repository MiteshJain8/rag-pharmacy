import httpx
import pytest

from app.db.repository import MedicineRepository
from app.services.rag import service as service_module
from app.services.rag.service import RagService


@pytest.mark.asyncio
async def test_clinical_advice_is_refused_before_database_access() -> None:
    result = await RagService().query("Can I replace my medicine with paracetamol?", 5, True)
    assert result.retrieval["safety_refusal"] is True
    assert result.sources == []
    assert "pharmacist" in result.answer


def test_catalog_pack_size_is_not_described_as_strength() -> None:
    source = {
        "source_id": "jan-aushadhi-1",
        "content": "unverified raw canonical text",
        "metadata": {
            "generic_name": "Aceclofenac 100mg and Paracetamol 325mg Tablets",
            "strength": "10's",
            "jan_aushadhi_mrp": 20,
            "source_url": "data/jan_aushadhi_products.csv",
            "source_date": "2026-08-29",
            "source_path": "private/path",
        },
    }
    public = RagService._public_source(source)
    assert "Pack size: 10's" in public["content"]
    assert "strength" not in public["metadata"]
    assert "source_url" not in public["metadata"]
    assert "source_path" not in public["metadata"]


def test_lexical_expansions_keep_highest_score_and_global_order() -> None:
    class Repository:
        def sparse_search(self, query: str, limit: int) -> list[dict]:
            if query == "first":
                return [
                    {"source_id": "a", "lexical_score": 0.1},
                    {"source_id": "b", "lexical_score": 0.3},
                ]
            return [
                {"source_id": "a", "lexical_score": 0.9},
                {"source_id": "c", "lexical_score": 0.5},
            ]

    results = RagService._sparse_search(Repository(), ["first", "second"])
    assert [row["source_id"] for row in results] == ["a", "c", "b"]
    assert results[0]["lexical_score"] == 0.9


def test_read_retries_one_transient_disconnect(monkeypatch) -> None:
    monkeypatch.setattr("app.db.repository.time.sleep", lambda _: None)

    class Request:
        calls = 0

        def execute(self):
            self.calls += 1
            if self.calls == 1:
                raise httpx.RemoteProtocolError("Server disconnected")
            return "ok"

    request = Request()
    assert MedicineRepository._execute_read(request) == "ok"
    assert request.calls == 2


@pytest.mark.asyncio
async def test_unrelated_query_abstains_on_incidental_word_match(monkeypatch) -> None:
    monkeypatch.setattr(service_module, "get_supabase_client", lambda: object())
    monkeypatch.setattr(RagService, "_embed", lambda self, query: [0.0])
    monkeypatch.setattr(
        RagService, "_dense_search", staticmethod(lambda repository, embeddings: [])
    )
    monkeypatch.setattr(
        RagService,
        "_sparse_search",
        staticmethod(
            lambda repository, queries: [
                {
                    "source_id": "cdsco-1",
                    "canonical_text": "Imported from France",
                    "lexical_score": 0.1,
                }
            ]
        ),
    )
    result = await RagService().query("What is the capital of France?", 5, False)
    assert result.sources == []
    assert result.retrieval["synthesis_mode"] == "abstain-no-lexical-match"
