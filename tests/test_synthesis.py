import pytest

from app.core.config import Settings
from app.services.rag.providers import GroqProvider
from app.services.rag.synthesis import GroundedSynthesizer


@pytest.mark.asyncio
async def test_answer_contains_only_displayed_record_with_source_id() -> None:
    synthesizer = GroundedSynthesizer(GroqProvider(Settings(groq_api_key="")))
    result = await synthesizer.synthesize(
        "paracetamol",
        [
            {
                "source_id": "jan-aushadhi-123",
                "content": "Paracetamol 500mg tablets | Pack size: 10's",
                "metadata": {"source_document": "Jan Aushadhi catalog snapshot"},
            }
        ],
    )
    assert result.mode == "source-extract"
    assert "[jan-aushadhi-123]" in result.answer
    assert "Paracetamol 500mg tablets" in result.answer
    assert "recommendation" in result.answer


@pytest.mark.asyncio
async def test_empty_evidence_abstains() -> None:
    result = await GroundedSynthesizer(GroqProvider(Settings(groq_api_key=""))).synthesize(
        "unknown medicine", []
    )
    assert "No matching source record" in result.answer
    assert result.mode == "source-extract"
