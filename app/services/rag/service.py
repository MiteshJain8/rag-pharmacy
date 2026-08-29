from __future__ import annotations

import asyncio
from functools import lru_cache

from fastembed import TextEmbedding

from app.core.config import Settings, get_settings
from app.db.client import get_supabase_client
from app.db.repository import MedicineRepository
from app.services.rag.canonicalize import validate_embedding
from app.services.rag.pipeline import RagPipeline, RetrievalResult
from app.services.rag.providers import CohereProvider, GroqProvider
from app.services.rag.synthesis import GroundedSynthesizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


@lru_cache
def get_embedding_model() -> TextEmbedding:
    return TextEmbedding(model_name=MODEL_NAME)


class RagService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.pipeline = RagPipeline()
        self.groq = GroqProvider(self.settings)
        self.cohere = CohereProvider(self.settings)
        self.synthesizer = GroundedSynthesizer(self.groq)

    async def query(self, query: str, top_k: int, expand_query: bool) -> RetrievalResult:
        expansion_failed = False
        if expand_query:
            try:
                queries = await self._expand(query)
            except Exception:
                queries = [query]
                expansion_failed = True
        else:
            queries = [query]
        repository = MedicineRepository(get_supabase_client())
        embeddings = await asyncio.gather(
            *(asyncio.to_thread(self._embed, expanded_query) for expanded_query in queries)
        )
        dense_results, sparse_results = await asyncio.gather(
            asyncio.to_thread(self._dense_search, repository, embeddings),
            asyncio.to_thread(self._sparse_search, repository, queries),
        )
        result = self.pipeline.build_context(
            dense_results, sparse_results, query=query, top_k=top_k
        )
        rerank_failed = False
        try:
            reranked_sources = await self._rerank(query, result.sources, top_k)
        except Exception:
            reranked_sources = result.sources
            rerank_failed = True
        result.sources = reranked_sources[:top_k]
        synthesis = await self.synthesizer.synthesize(query, result.sources)
        result.answer = synthesis.answer
        result.retrieval["expanded_queries"] = queries
        result.retrieval["dense_queries"] = len(embeddings)
        result.retrieval["expanded"] = len(queries) > 1
        result.retrieval["expansion_failed"] = expansion_failed
        result.retrieval["reranked"] = self.cohere.api_key != "" and not rerank_failed
        result.retrieval["rerank_failed"] = rerank_failed
        result.retrieval["synthesis_mode"] = synthesis.mode
        return result

    async def _expand(self, query: str) -> list[str]:
        expanded = await self.groq.complete(
            "Return up to three concise search rewrites, one per line. "
            "Preserve exact medicine names, salts, strengths, and units. "
            f"Query: {query}"
        )
        if not expanded:
            return [query]
        rewrites = [line.strip(" -") for line in expanded.splitlines() if line.strip()]
        return list(dict.fromkeys([query, *rewrites[:3]]))

    def _embed(self, query: str) -> list[float]:
        embedding = next(get_embedding_model().embed([query])).tolist()
        validate_embedding(embedding)
        return embedding

    @staticmethod
    def _dense_search(repository: MedicineRepository, embeddings: list[list[float]]) -> list[dict]:
        results: dict[str, dict] = {}
        for embedding in embeddings:
            for item in repository.dense_search(embedding, 25):
                existing = results.get(item["source_id"])
                if existing is None or item.get("similarity", 0) > existing.get("similarity", 0):
                    results[item["source_id"]] = item
        return sorted(results.values(), key=lambda item: item.get("similarity", 0), reverse=True)[
            :25
        ]

    @staticmethod
    def _sparse_search(repository: MedicineRepository, queries: list[str]) -> list[dict]:
        results: dict[str, dict] = {}
        for query in queries:
            for item in repository.sparse_search(query, 25):
                results.setdefault(item["source_id"], item)
        return list(results.values())[:25]

    async def _rerank(self, query: str, sources: list[dict], top_k: int) -> list[dict]:
        reranked = await self.cohere.rerank(query, [source["content"] for source in sources], top_k)
        if reranked is None:
            return sources
        selected = []
        for item in reranked:
            if item.index < len(sources):
                source = sources[item.index]
                source["metadata"]["cohere_relevance_score"] = item.score
                selected.append(source)
        return selected
