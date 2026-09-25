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
                "No matching source record could be identified unambiguously. "
                "This does not establish whether "
                "a medicine is safe or available.",
                "source-extract",
            )
        source = sources[0]
        source_id = source["source_id"]
        content = source["content"]
        metadata = source.get("metadata", {})
        if source_id.startswith("cdsco-"):
            entry = content.split("|", 1)[0].removeprefix("Banned Drug Entry ").strip()
            answer = f"The CDSCO banned-drug list records {entry} [{source_id}]"
        elif source_id.startswith("jan-aushadhi-"):
            answer = f"The Jan Aushadhi catalog snapshot lists {content}. [{source_id}]"
        elif source_id.startswith("kendra-"):
            answer = f"The Karnataka Jan Aushadhi Kendra directory lists {content}. [{source_id}]"
        else:
            name = metadata.get("brand_name") or metadata.get("generic_name") or "This product"
            answer = f"An OpenFDA label record lists {name}. [{source_id}]"
        return SynthesisResult(answer, "single-source")
