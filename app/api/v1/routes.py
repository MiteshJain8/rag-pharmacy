from fastapi import APIRouter, Depends, HTTPException

from app.api.v1.schemas import QueryRequest, QueryResponse
from app.services.rag.service import RagService

router = APIRouter(prefix="/api/v1", tags=["query"])


def get_rag_service() -> RagService:
    return RagService()


@router.post("/query", response_model=QueryResponse)
async def query_medicines(
    request: QueryRequest, service: RagService = Depends(get_rag_service)
) -> QueryResponse:
    try:
        result = await service.query(request.query, request.top_k, request.expand_query)
    except RuntimeError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    return QueryResponse(
        answer=result.answer,
        confidence=max((source["confidence"] for source in result.sources), default=0.0),
        sources=result.sources,
        retrieval={"query": request.query, **result.retrieval},
    )
