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
    assert result.mode == "single-source"
    assert "[jan-aushadhi-123]" in result.answer
    assert "Paracetamol 500mg tablets" in result.answer
    assert result.answer.count("[jan-aushadhi-123]") == 1


@pytest.mark.asyncio
async def test_answer_does_not_append_unrelated_records() -> None:
    synthesizer = GroundedSynthesizer(GroqProvider(Settings(groq_api_key="")))
    result = await synthesizer.synthesize(
        "CDSCO banned Phenacetin",
        [
            {
                "source_id": "cdsco-banned-8",
                "content": "Banned Drug Entry #8: Phenacetin. | Gazette Notification",
                "metadata": {},
            },
            {
                "source_id": "cdsco-banned-289",
                "content": "Banned Drug Entry #289: Dextromethorphan",
                "metadata": {},
            },
        ],
    )
    assert "Phenacetin" in result.answer
    assert "Dextromethorphan" not in result.answer
    assert "cdsco-banned-289" not in result.answer


@pytest.mark.asyncio
async def test_empty_evidence_abstains() -> None:
    result = await GroundedSynthesizer(GroqProvider(Settings(groq_api_key=""))).synthesize(
        "unknown medicine", []
    )
    assert "No matching source record" in result.answer
    assert result.mode == "source-extract"
