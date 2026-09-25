from __future__ import annotations

from dataclasses import dataclass

import httpx

from app.core.config import Settings


@dataclass(frozen=True)
class RerankResult:
    index: int
    score: float


class GroqProvider:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.groq_api_key
        self.model = settings.groq_model
        self.last_usage: dict = {}

    async def complete(self, prompt: str) -> str | None:
        if not self.api_key:
            return None
        headers = {"Authorization": f"Bearer {self.api_key}"}
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 500,
        }
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=headers,
                json=payload,
            )
            response.raise_for_status()
        body = response.json()
        self.last_usage = body.get("usage", {})
        return body["choices"][0]["message"]["content"]


class CohereProvider:
    def __init__(self, settings: Settings) -> None:
        self.api_key = settings.cohere_api_key
        self.model = settings.cohere_rerank_model
        self.last_usage: dict = {}

    async def rerank(
        self, query: str, documents: list[str], top_n: int
    ) -> list[RerankResult] | None:
        if not self.api_key or not documents:
            return None
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        payload = {"model": self.model, "query": query, "documents": documents, "top_n": top_n}
        async with httpx.AsyncClient(timeout=20) as client:
            response = await client.post(
                "https://api.cohere.com/v2/rerank", headers=headers, json=payload
            )
            response.raise_for_status()
        body = response.json()
        self.last_usage = body.get("meta", {}).get("billed_units", {})
        return [
            RerankResult(index=item["index"], score=float(item["relevance_score"]))
            for item in body["results"]
        ]
