import pytest

from scripts.ingest_pdfs import chunk_text


def test_chunk_text_uses_overlap() -> None:
    chunks = chunk_text("one two three four five six", chunk_size=4, overlap=1)
    assert chunks == ["one two three four", "four five six"]


def test_chunk_text_rejects_invalid_overlap() -> None:
    with pytest.raises(ValueError, match="overlap"):
        chunk_text("text", chunk_size=4, overlap=4)
