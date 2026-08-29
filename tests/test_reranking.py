import pytest

from app.core.config import Settings
from app.services.rag.providers import RerankResult
from app.services.rag.service import RagService


@pytest.mark.asyncio
async def test_rerank_receives_candidate_pool_and_returns_top_five() -> None:
    service = RagService(Settings())
    received: list[str] = []

    async def fake_rerank(query: str, documents: list[str], top_n: int) -> list[RerankResult]:
        received.extend(documents)
        assert top_n == 5
        return [RerankResult(index=index, score=0.9 - index / 100) for index in range(5)]

    service.cohere.rerank = fake_rerank
    sources = [
        {"source_id": str(index), "content": f"record {index}", "metadata": {}}
        for index in range(50)
    ]

    reranked = await service._rerank("paracetamol", sources, 5)

    assert len(received) == 50
    assert [source["source_id"] for source in reranked] == ["0", "1", "2", "3", "4"]
    assert reranked[0]["metadata"]["cohere_relevance_score"] == 0.9
