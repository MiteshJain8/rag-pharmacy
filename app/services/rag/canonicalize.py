import re


def canonicalize_medicine_text(
    *,
    generic_name: str,
    brand_name: str,
    manufacturer: str,
    strength: str,
    dosage_form: str,
    contraindications: str,
) -> str:
    fields = {
        "generic salt": generic_name,
        "brand": brand_name,
        "manufacturer": manufacturer,
        "strength": strength,
        "dosage form": dosage_form,
        "contraindications": contraindications,
    }
    normalized = []
    for label, value in fields.items():
        value = re.sub(r"[^\w./+% -]", " ", value, flags=re.UNICODE)
        value = re.sub(r"\s+", " ", value).strip()
        normalized.append(f"{label}: {value}")
    return " | ".join(normalized)


def validate_embedding(embedding: list[float], expected_dimension: int = 384) -> None:
    if len(embedding) != expected_dimension:
        raise ValueError(
            f"embedding dimension mismatch: expected {expected_dimension}, got {len(embedding)}"
        )
