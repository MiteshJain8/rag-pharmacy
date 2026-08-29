from app.services.rag.pipeline import RagPipeline


def test_pipeline_fuses_and_limits_sources() -> None:
    dense = [{"source_id": f"d-{index}"} for index in range(30)]
    sparse = [{"source_id": "d-0"}] + [{"source_id": f"s-{index}"} for index in range(29)]

    result = RagPipeline().build_context(dense, sparse, query="paracetamol")

    assert len(result.sources) == 49
    assert result.sources[0]["source_id"] == "d-0"
    assert result.sources[0]["metadata"]["dense_rank"] == 1
    assert result.sources[0]["metadata"]["sparse_rank"] == 1
    assert result.retrieval["candidate_pool_limit"] == 50
    assert result.retrieval["rrf_k"] == 60
