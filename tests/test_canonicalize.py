import pytest

from app.services.rag.canonicalize import canonicalize_medicine_text, validate_embedding


def test_canonical_text_preserves_compound_salt_and_normalizes_spacing() -> None:
    text = canonicalize_medicine_text(
        generic_name="Amoxicillin + Clavulanic Acid",
        brand_name="Augmentin 625",
        manufacturer="Example Ltd.",
        strength="625 mg",
        dosage_form="film-coated tablet",
        contraindications="Penicillin hypersensitivity",
    )
    assert "Amoxicillin + Clavulanic Acid" in text
    assert "  " not in text


def test_embedding_dimension_is_checked() -> None:
    validate_embedding([0.0] * 384)
    with pytest.raises(ValueError, match="dimension"):
        validate_embedding([0.0] * 383)
