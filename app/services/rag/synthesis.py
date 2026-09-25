from __future__ import annotations

from dataclasses import dataclass

from app.services.rag.providers import GroqProvider


@dataclass(frozen=True)
class SynthesisResult:
    answer: str
    mode: str


class GroundedSynthesizer:
    """Public demo uses extractive answers so every displayed claim has a source ID."""

    def __init__(self, provider: GroqProvider) -> None:
        self.provider = provider

    async def synthesize(self, query: str, sources: list[dict]) -> SynthesisResult:
        if not sources:
            return SynthesisResult(
                "No matching source record was found. This does not establish whether "
                "a medicine is safe or available.",
                "source-extract",
            )
        lines = ["Matching source records (not a substitution or treatment recommendation):"]
        for source in sources:
            metadata = source.get("metadata", {})
            label = metadata.get("source_document", "Source record")
            page = metadata.get("page_number")
            page_label = f", p. {page}" if page else ""
            lines.append(f"- [{source['source_id']}] {source['content']} ({label}{page_label})")
        lines.append("Verify the source and current details with a pharmacist or doctor.")
        return SynthesisResult("\n".join(lines), "source-extract")
