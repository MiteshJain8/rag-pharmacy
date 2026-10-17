from __future__ import annotations

import re
from dataclasses import dataclass

from app.services.rag.providers import GroqProvider


@dataclass(frozen=True)
class SynthesisResult:
    answer: str
    mode: str


class GroundedSynthesizer:
    """Only a locally verified rewrite may replace the source-backed template."""

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
        if self.provider.api_key and not _instruction_like(answer):
            try:
                rewritten = await self.provider.format_answer(answer)
                if _valid_rewrite(rewritten, answer, source_id):
                    return SynthesisResult(rewritten, "groq-validated")
            except Exception:
                pass
        return SynthesisResult(answer, "single-source")


def _instruction_like(value: str) -> bool:
    return bool(re.search(
        r"(?i)\b(ignore (all |any )?(previous|prior|above) instructions|"
        r"system prompt|developer message|you are now|act as|jailbreak|"
        r"disregard (the )?(rules|instructions)|override (the )?(rules|instructions))\b",
        value,
    ))


def _valid_rewrite(candidate: str | None, fact: str, source_id: str) -> bool:
    if not isinstance(candidate, str) or len(candidate) > 350:
        return False
    if candidate.count(f"[{source_id}]") != 1 or _instruction_like(candidate):
        return False
    if any(token in candidate for token in ("\n", "http://", "https://", "<", ">")):
        return False
    # New substantive words can introduce unsupported claims. Fall back to the vetted fact.
    function_words = {"a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "of",
                      "on", "the", "to", "was", "with"}
    def words(text: str) -> set[str]:
        return set(re.findall(r"[a-z0-9]+", text.lower()))
    return bool(candidate.strip()) and words(candidate) <= words(fact) | function_words
