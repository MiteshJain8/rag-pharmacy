import pytest

from app.core.config import Settings
from app.services.rag.providers import GroqProvider
from app.services.rag.synthesis import GroundedSynthesizer


@pytest.mark.asyncio
async def test_synthesis_fallback_is_grounded_in_source_metadata() -> None:
    synthesizer = GroundedSynthesizer(GroqProvider(Settings(groq_api_key="")))
    result = await synthesizer.synthesize(
        "paracetamol substitute",
        [
            {
                "content": "generic salt: Paracetamol",
                "metadata": {
                    "generic_name": "Paracetamol",
                    "brand_name": "Calpol 500",
                    "strength": "500 mg",
                    "dosage_form": "tablet",
                    "jan_aushadhi_mrp": 0.5,
                    "brand_mrp": 2.2,
                    "contraindications": "Severe liver disease",
                },
            }
        ],
    )

    assert result.mode == "deterministic"
    assert "Calpol 500" in result.answer
    assert "Severe liver disease" in result.answer


@pytest.mark.asyncio
async def test_synthesis_reports_empty_evidence() -> None:
    synthesizer = GroundedSynthesizer(GroqProvider(Settings(groq_api_key="")))
    result = await synthesizer.synthesize("unknown medicine", [])
    assert "No medicine records matched" in result.answer


@pytest.mark.asyncio
async def test_synthesis_falls_back_when_groq_fails() -> None:
    class FailingProvider:
        api_key = "configured"

        async def complete(self, prompt: str) -> str | None:
            raise RuntimeError("provider unavailable")

    result = await GroundedSynthesizer(FailingProvider()).synthesize(
        "paracetamol",
        [{"content": "generic salt: Paracetamol", "metadata": {"generic_name": "Paracetamol"}}],
    )

    assert result.mode == "deterministic-fallback"
    assert "Paracetamol" in result.answer
