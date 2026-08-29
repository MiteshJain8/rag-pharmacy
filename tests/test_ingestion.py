from pathlib import Path

from scripts.ingest_data import build_rows, load_jan_aushadhi_csv, normalize_records


def test_fixture_rows_have_stable_search_text() -> None:
    rows = build_rows()
    assert len(rows) == 4
    assert {row["source_id"] for row in rows} == {
        "fixture-paracetamol-500",
        "fixture-amlodipine-5",
        "fixture-metformin-500",
        "fixture-omeprazole-20",
    }
    assert all(row["canonical_text"] for row in rows)


def test_external_records_accept_common_dataset_aliases() -> None:
    rows = normalize_records(
        [
            {
                "salt": "Metformin Hydrochloride",
                "brand": "Glycomet 500",
                "company": "USV",
                "dose": "500 mg",
                "form": "tablet",
                "mrp": "4.80",
                "source": "community-dataset",
            }
        ],
        "community-dataset",
    )

    assert len(rows) == 1
    assert rows[0]["generic_name"] == "Metformin Hydrochloride"
    assert rows[0]["brand_mrp"] == 4.8
    assert len(rows[0]["source_id"]) == 24


def test_jan_aushadhi_csv_maps_each_catalog_column() -> None:
    rows = load_jan_aushadhi_csv(Path("data/jan_aushadhi_products.csv"))
    assert len(rows) == 2439
    assert rows[0]["source_id"] == "jan-aushadhi-1"
    assert rows[0]["drug_code"] == "1"
    assert rows[0]["unit_size"] == "10's"
    assert rows[0]["therapeutic_group"] == "Analgesic/Antipyretic/Anti-Inflammatory"
    assert rows[0]["generic_name"].startswith("Aceclofenac")
    assert rows[0]["strength"] == "10's"
    assert rows[0]["dosage_form"] == "Analgesic/Antipyretic/Anti-Inflammatory"
    assert rows[0]["jan_aushadhi_mrp"] == 10.32
