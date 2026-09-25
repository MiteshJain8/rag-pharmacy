from fastapi import APIRouter, Depends, HTTPException, Request

from app.api.v1.rate_limit import query_limiter
from app.api.v1.schemas import QueryRequest, QueryResponse
from app.services.rag.service import RagService

router = APIRouter(prefix="/api/v1", tags=["query"])


def get_rag_service() -> RagService:
    return RagService()


@router.post("/query", response_model=QueryResponse)
async def query_medicines(
    request: QueryRequest, http_request: Request, service: RagService = Depends(get_rag_service)
) -> QueryResponse:
    if not query_limiter.allow(http_request.client.host if http_request.client else "unknown"):
        raise HTTPException(status_code=429, detail="Too many queries. Please try again shortly.")
    try:
        result = await service.query(request.query, request.top_k, request.expand_query)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return QueryResponse(
        answer=result.answer,
        sources=result.sources,
        retrieval={"query": request.query, **result.retrieval},
    )
