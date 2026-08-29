from __future__ import annotations

from dataclasses import dataclass

from app.services.rag.providers import GroqProvider


@dataclass(frozen=True)
class SynthesisResult:
    answer: str
    mode: str


class GroundedSynthesizer:
    def __init__(self, provider: GroqProvider) -> None:
        self.provider = provider

    async def synthesize(self, query: str, sources: list[dict]) -> SynthesisResult:
        fallback = self._fallback(query, sources)
        if not self.provider.api_key or not sources:
            return SynthesisResult(fallback, "deterministic")

        evidence = "\n\n".join(
            f"[{index}] {source['content']}" for index, source in enumerate(sources, 1)
        )
        prompt = (
            """You are the specialized AI Assistant for the Jan Aushadhi & CDSCO Pharma Knowledge Base.
            STRICT OPERATIONAL RULES:
            1. Grounding Only: Answer the question STRICTLY using ONLY the facts present in the provided context chunks below. Do NOT use outside medical knowledge or assumptions.
            2. Unanswerable / Missing Information: If the provided context does not contain the exact facts needed to answer the question, respond with:
            "I cannot find sufficient information in the Jan Aushadhi catalog, CDSCO banned drugs register, or Kendra directory to answer this question."
            3. Irrelevant / Out-of-Domain Questions: If the user asks anything unrelated to Indian pharmaceuticals, Jan Aushadhi medicines/pricing, CDSCO regulations, or Karnataka Kendras (e.g., coding, general chat, sports), immediately decline:
            "I can only assist with queries related to Jan Aushadhi medicines, PMBI pricing, CDSCO prohibited drugs, and Karnataka Kendras."
            4. Zero-MRP Handling: If a product has an MRP of 0.00 or null, explicitly state that its price is currently "Under Process / Under Rate Revision by PMBI".
            5. Citations: Always cite the source document name, Drug Code, or Gazette Notification if available.
            """
            f"User question: {query}\n\nEvidence:\n{evidence}"
        )
        try:
            answer = await self.provider.complete(prompt)
        except Exception:
            return SynthesisResult(fallback, "deterministic-fallback")
        return SynthesisResult(answer or fallback, "groq" if answer else "deterministic")

    @staticmethod
    def _fallback(query: str, sources: list[dict]) -> str:
        if not sources:
            return f"No medicine records matched '{query}'. Consult a pharmacist for alternatives."

        lines = [f"Retrieved medicine information for '{query}':"]
        for source in sources:
            metadata = source.get("metadata", {})
            price_text = GroundedSynthesizer._price_text(metadata)
            contraindications = metadata.get("contraindications") or "Not provided"
            generic_name = metadata.get("generic_name", "Unknown salt")
            brand_name = metadata.get("brand_name", "Unknown brand")
            strength = metadata.get("strength", "strength unavailable")
            dosage_form = metadata.get("dosage_form", "")
            lines.append(
                f"- {generic_name} ({brand_name}), {strength} {dosage_form}; "
                f"{price_text}; contraindications: {contraindications}."
            )
        lines.append(
            "This is reference information, not a diagnosis or prescription; "
            "consult a pharmacist or doctor."
        )
        return "\n".join(lines)

    @staticmethod
    def _price_text(metadata: dict) -> str:
        jan_price = metadata.get("jan_aushadhi_mrp")
        brand_price = metadata.get("brand_mrp")
        if jan_price is None and brand_price is None:
            return "prices unavailable"
        return f"Jan Aushadhi MRP: INR {jan_price}; brand MRP: INR {brand_price}"
