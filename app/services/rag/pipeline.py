from collections.abc import Sequence
from dataclasses import dataclass

from app.services.rag.rrf import RankedItem, reciprocal_rank_fusion

RRF_K = 60
RETRIEVAL_LIMIT = 25
CONTEXT_LIMIT = 5


@dataclass
class RetrievalResult:
    answer: str
    sources: list[dict]
    retrieval: dict


class RagPipeline:
    """Provider-neutral orchestration boundary for dense, lexical, and reranked evidence."""

    def build_context(
        self,
        dense_results: Sequence[dict],
        sparse_results: Sequence[dict],
        *,
        query: str,
        top_k: int = CONTEXT_LIMIT,
    ) -> RetrievalResult:
        dense = [
            RankedItem(item["source_id"], rank, item)
            for rank, item in enumerate(dense_results[:RETRIEVAL_LIMIT], 1)
        ]
        sparse = [
            RankedItem(item["source_id"], rank, item)
            for rank, item in enumerate(sparse_results[:RETRIEVAL_LIMIT], 1)
        ]
        dense_ranks = {item.item_id: item.rank for item in dense}
        sparse_ranks = {item.item_id: item.rank for item in sparse}
        merged_payloads = {item.item_id: dict(item.payload) for item in dense}
        for item in sparse:
            merged_payloads.setdefault(item.item_id, {}).update(item.payload)
        for item_id, payload in merged_payloads.items():
            payload["dense_rank"] = dense_ranks.get(item_id)
            payload["sparse_rank"] = sparse_ranks.get(item_id)
            payload["dense_similarity"] = next(
                (item.payload.get("similarity") for item in dense if item.item_id == item_id), None
            )
            payload["sparse_lexical_score"] = next(
                (item.payload.get("lexical_score") for item in sparse if item.item_id == item_id),
                None,
            )
        fused = reciprocal_rank_fusion([dense, sparse], k=RRF_K, limit=50)
        sources = [
            {
                "source_id": item_id,
                "content": payload.get("canonical_text", ""),
                "confidence": min(1.0, score * (RRF_K + 1)),
                "metadata": {
                    **{
                        key: value
                        for key, value in merged_payloads[item_id].items()
                        if key != "source_id"
                    },
                    "rrf_score": score,
                    "rrf_rank": rank,
                },
            }
            for rank, (item_id, score, payload) in enumerate(fused, 1)
        ]
        answer = (
            f"No synthesis provider is configured. Retrieved {len(sources)} evidence records "
            f"for: {query}"
        )
        return RetrievalResult(
            answer=answer,
            sources=sources,
            retrieval={
                "rrf_k": RRF_K,
                "dense_limit": RETRIEVAL_LIMIT,
                "sparse_limit": RETRIEVAL_LIMIT,
                "fused_limit": 50,
                "candidate_pool_limit": 50,
                "context_limit": top_k,
            },
        )
