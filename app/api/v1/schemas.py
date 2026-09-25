from typing import Any

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    query: str = Field(min_length=2, max_length=500)
    top_k: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Optional reranker shortlist size. The response contains at most one source.",
    )
    expand_query: bool = False

    @field_validator("query")
    @classmethod
    def normalize_query(cls, value: str) -> str:
        normalized = " ".join(value.split())
        if not normalized:
            raise ValueError("query must contain non-whitespace characters")
        return normalized


class SourceChunk(BaseModel):
    source_id: str
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    retrieval: dict[str, Any] = Field(default_factory=dict)
