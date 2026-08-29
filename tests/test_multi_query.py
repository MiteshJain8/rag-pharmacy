from app.services.rag.service import RagService


class FakeRepository:
    def __init__(self) -> None:
        self.calls: list[list[float]] = []

    def dense_search(self, embedding: list[float], limit: int) -> list[dict]:
        self.calls.append(embedding)
        return [{"source_id": str(len(self.calls)), "similarity": float(len(self.calls))}]


def test_dense_search_merges_results_from_all_expanded_queries() -> None:
    repository = FakeRepository()
    results = RagService._dense_search(repository, [[1.0], [2.0], [3.0]])

    assert len(repository.calls) == 3
    assert [item["source_id"] for item in results] == ["3", "2", "1"]
