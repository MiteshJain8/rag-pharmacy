from __future__ import annotations

import asyncio
import os
import re
import time
from functools import lru_cache

from fastembed import TextEmbedding

from app.core.config import Settings, get_settings
from app.db.client import get_readonly_supabase_client
from app.db.repository import MedicineRepository
from app.services.rag.canonicalize import validate_embedding
from app.services.rag.pipeline import RagPipeline, RetrievalResult
from app.services.rag.providers import CohereProvider, GroqProvider
from app.services.rag.selection import select_source
from app.services.rag.synthesis import GroundedSynthesizer

MODEL_NAME = "BAAI/bge-small-en-v1.5"
CLINICAL_REQUEST = re.compile(
    r"\b(should i (take|stop|switch|replace)|can i (take|switch|replace)|"
    r"what (dose|dosage)|how much (should i|can i) take|prescri(be|ption)|"
    r"safe (for me|during pregnancy)|best substitute|equivalent medicine|"
    r"treat my|diagnos(e|is))\b",
    re.IGNORECASE,
)
DOMAIN_MARKER = re.compile(
    r"\b(cdsco|jan aushadhi|kendra|medicine|drug|tablet|capsule|gel|injection|"
    r"label|mg|mcg|pharmacy|ban(?:ned)?|PMBJK\d+)\b",
    re.IGNORECASE,
)
STOP_WORDS = {
    "the",
    "and",
    "for",
    "what",
    "this",
    "that",
    "with",
    "from",
    "into",
    "about",
    "how",
    "who",
    "is",
    "are",
    "can",
    "you",
    "me",
    "tell",
    "write",
    "code",
}


@lru_cache
def get_embedding_model() -> TextEmbedding:
    return TextEmbedding(model_name=MODEL_NAME, cache_dir=os.getenv("FASTEMBED_CACHE_DIR"))


class RagService:
    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self.pipeline = RagPipeline()
        self.groq = GroqProvider(self.settings)
        self.cohere = CohereProvider(self.settings)
        self.synthesizer = GroundedSynthesizer(self.groq)

    async def query(self, query: str, top_k: int, expand_query: bool) -> RetrievalResult:
        started = time.perf_counter()
        if CLINICAL_REQUEST.search(query):
            return RetrievalResult(
                answer=(
                    "I can show source records, but I cannot advise on taking, changing, "
                    "or substituting medicines. Please ask a pharmacist or doctor."
                ),
                sources=[],
                retrieval={"safety_refusal": True, "elapsed_ms": 0},
            )
        expansion_failed = False
        if expand_query:
            try:
                queries = await self._expand(query)
            except Exception:
                queries = [query]
                expansion_failed = True
        else:
            queries = [query]
        expansion_ms = round((time.perf_counter() - started) * 1000, 1)
        repository = MedicineRepository(get_readonly_supabase_client())
        embeddings = await asyncio.gather(
            *(asyncio.to_thread(self._embed, expanded_query) for expanded_query in queries)
        )
        dense_results, sparse_results = await asyncio.gather(
            asyncio.to_thread(self._dense_search, repository, embeddings),
            asyncio.to_thread(self._sparse_search, repository, queries),
        )
        retrieval_ms = round((time.perf_counter() - started) * 1000 - expansion_ms, 1)
        dense_results = [
            row for row in dense_results if not row["source_id"].startswith("fixture-")
        ]
        sparse_results = [
            row for row in sparse_results if not row["source_id"].startswith("fixture-")
        ]
        kendra_code = re.search(r"\bPMBJK\d+\b", query, re.IGNORECASE)
        if kendra_code:
            exact_id = f"kendra-karnataka-{kendra_code.group().upper()}"
            exact = await asyncio.to_thread(repository.lookup_chunk, exact_id)
            if exact:
                sparse_results = [
                    exact,
                    *[r for r in sparse_results if r["source_id"] != exact_id],
                ][:25]
        category_prefix = None
        if re.search(r"\bCDSCO\b", query, re.IGNORECASE):
            category_prefix = "cdsco-"
            focused = await asyncio.to_thread(repository.search_cdsco_terms, query)
            sparse_results = [
                *focused,
                *[
                    r
                    for r in sparse_results
                    if r["source_id"] not in {f["source_id"] for f in focused}
                ],
            ][:25]
        elif re.search(r"\bKendra\b|\bPMBJK\d+\b", query, re.IGNORECASE):
            category_prefix = "kendra-"
        if category_prefix:
            dense_results = [r for r in dense_results if r["source_id"].startswith(category_prefix)]
            sparse_results = [
                r for r in sparse_results if r["source_id"].startswith(category_prefix)
            ]
        informative = {
            token
            for token in re.findall(r"[a-z0-9]+", query.lower())
            if len(token) >= 3 and token not in STOP_WORDS
        }
        best_overlap = max(
            (
                len(
                    informative & set(re.findall(r"[a-z0-9]+", r.get("canonical_text", "").lower()))
                )
                for r in sparse_results[:10]
            ),
            default=0,
        )
        if not sparse_results or (not DOMAIN_MARKER.search(query) and best_overlap < 2):
            return RetrievalResult(
                answer="No matching source record was found. This does not establish whether "
                "a medicine is safe or available.",
                sources=[],
                retrieval={
                    "expanded_queries": queries,
                    "synthesis_mode": "abstain-no-lexical-match",
                    "timing_ms": {"total": round((time.perf_counter() - started) * 1000, 1)},
                    "provider_usage": {"groq": self.groq.last_usage, "cohere": {}},
                },
            )
        result = self.pipeline.build_context(
            dense_results, sparse_results, query=query, top_k=top_k
        )
        rerank_failed = False
        rerank_started = time.perf_counter()
        try:
            reranked_sources = await self._rerank(query, result.sources, top_k)
        except Exception:
            reranked_sources = result.sources
            rerank_failed = True
        if category_prefix and result.sources:
            first = result.sources[0]
            reranked_sources = [
                first,
                *[s for s in reranked_sources if s["source_id"] != first["source_id"]],
            ]
        selected = select_source(query, reranked_sources)
        result.sources = [selected] if selected else []
        rerank_ms = round((time.perf_counter() - rerank_started) * 1000, 1)
        result.sources = [self._public_source(source) for source in result.sources]
        synthesis = await self.synthesizer.synthesize(query, result.sources)
        result.answer = synthesis.answer
        result.retrieval["expanded_queries"] = queries
        result.retrieval["dense_queries"] = len(embeddings)
        result.retrieval["expanded"] = len(queries) > 1
        result.retrieval["expansion_failed"] = expansion_failed
        result.retrieval["reranked"] = self.cohere.api_key != "" and not rerank_failed
        result.retrieval["rerank_failed"] = rerank_failed
        result.retrieval["synthesis_mode"] = synthesis.mode
        result.retrieval["timing_ms"] = {
            "expansion": expansion_ms,
            "retrieval_and_embedding": retrieval_ms,
            "rerank": rerank_ms,
            "total": round((time.perf_counter() - started) * 1000, 1),
        }
        result.retrieval["provider_usage"] = {
            "groq": self.groq.last_usage,
            "cohere": self.cohere.last_usage,
        }
        return result

    @staticmethod
    def _public_source(source: dict) -> dict:
        metadata = dict(source["metadata"])
        source_id = source["source_id"]
        metadata.pop("canonical_text", None)
        metadata.pop("source_path", None)
        url = metadata.get("source_url", "")
        if not str(url).startswith(("https://", "http://")):
            metadata.pop("source_url", None)
        if source_id.startswith("jan-aushadhi-"):
            pack_size = metadata.pop("strength", "")
            metadata["pack_size"] = pack_size
            metadata["source_document"] = "Jan Aushadhi catalog snapshot"
            metadata["catalog_import_date"] = metadata.pop("source_date", None)
            price = metadata.get("jan_aushadhi_mrp")
            content = (
                f"{metadata.get('generic_name', 'Product')}"
                f" | Pack size: {pack_size or 'not listed'}"
                f" | Recorded MRP: INR {price if price is not None else 'unavailable'}"
            )
        elif source_id.startswith("cdsco-"):
            metadata["source_document"] = "CDSCO banned drug list"
            content = source["content"]
        elif source_id.startswith("kendra-"):
            metadata["source_document"] = "Karnataka Jan Aushadhi Kendra directory"
            content = source["content"]
        else:
            metadata["source_document"] = "OpenFDA label record"
            content = (
                f"{metadata.get('generic_name', 'Medicine')} | "
                f"Brand: {metadata.get('brand_name', 'not listed')} | "
                f"Warnings: {str(metadata.get('contraindications') or 'not provided')[:300]}"
            )
        return {"source_id": source_id, "content": content, "metadata": metadata}

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
                existing = results.get(item["source_id"])
                if existing is None or item.get("lexical_score", 0) > existing.get(
                    "lexical_score", 0
                ):
                    results[item["source_id"]] = item
        return sorted(
            results.values(), key=lambda item: item.get("lexical_score", 0), reverse=True
        )[:25]

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
