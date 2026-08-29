from pathlib import Path

from scripts.ingest_pdfs import extract_cdsco_rows, extract_kendra_rows


def test_cdsco_rows_keep_entry_boundaries() -> None:
    rows = extract_cdsco_rows(Path("data/cdsco_banned_drugs.pdf"))
    assert len(rows) == 444
    assert rows[0]["content"].startswith("Banned Drug Entry #1:")
    assert "Prohibited under Gazette Notification:" in rows[0]["content"]
    assert "\n" not in rows[0]["content"]


def test_kendra_rows_keep_columns_in_place() -> None:
    rows = extract_kendra_rows(Path("data/kendra_karnataka.pdf"))
    assert len(rows) == 1760
    assert rows[0]["content"].startswith("Jan Aushadhi Kendra PMBJK00884:")
    assert "District: Chikkaballapura" in rows[0]["content"]
    assert "Pincode: 561207" in rows[0]["content"]
    assert "Address: General Hospital Compound" in rows[0]["content"]
    assert "\n" not in rows[0]["content"]
