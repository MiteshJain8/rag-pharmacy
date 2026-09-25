"""Choose one directly matching record from a ranked retrieval pool."""

from __future__ import annotations

import re

QUERY_WORDS = {
    "aushadhi", "ban", "banned", "cdsco", "drug", "entry", "jan", "kendra",
    "label", "list", "medicine", "notification", "record", "source", "the",
    "what", "where", "which", "with", "combination", "systemic", "use",
}


def _terms(value: str) -> set[str]:
    return {word for word in re.findall(r"[a-z0-9]+", value.lower()) if len(word) >= 3}


def _compact(value: str) -> str:
    return "".join(re.findall(r"[a-z0-9]+", value.lower()))


def select_source(query: str, sources: list[dict]) -> dict | None:
    """Prefer a direct name/code match, using retrieval order only to break ties."""
    wanted = _terms(query) - QUERY_WORDS
    if not wanted:
        return None
    best: dict | None = None
    best_score = (0, 0.0, float("-inf"))
    single_term_cdsco_matches = 0
    for source in sources:
        source_id = source["source_id"]
        content = source.get("content", "")
        is_cdsco = source_id.startswith("cdsco-")
        if is_cdsco:
            # Gazette text is repeated across entries and must not count as a drug match.
            content = content.split("|", 1)[0].split(":", 1)[-1]
        metadata = source.get("metadata", {})
        generic_name = str(metadata.get("generic_name") or "")
        brand_name = str(metadata.get("brand_name") or "")
        searchable = " ".join((source_id, content, generic_name, brand_name))
        terms = _terms(searchable)
        compact = _compact(searchable)
        matched = {word for word in wanted if word in terms or (is_cdsco and word in compact)}
        if is_cdsco and len(wanted) == 1 and matched:
            single_term_cdsco_matches += 1
        exact_name = int(bool(generic_name) and _compact(query) == _compact(generic_name))
        extra_ingredients = len(_terms(content) - wanted - QUERY_WORDS) if is_cdsco else 0
        score = (exact_name, len(matched) / len(wanted), -extra_ingredients)
        if score > best_score:
            best, best_score = source, score
    threshold = 1.0 if len(wanted) == 1 else 0.6
    if single_term_cdsco_matches > 1:
        return None
    return best if best_score[1] >= threshold else None
