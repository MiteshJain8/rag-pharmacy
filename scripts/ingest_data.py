from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import httpx

from app.services.rag.canonicalize import canonicalize_medicine_text, validate_embedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"


@dataclass(frozen=True)
class MedicineFixture:
    source_id: str
    generic_name: str
    brand_name: str
    manufacturer: str
    strength: str
    dosage_form: str
    jan_aushadhi_mrp: float | None
    brand_mrp: float | None
    contraindications: str
    source_url: str
    source_date: date


FIXTURES = (
    MedicineFixture(
        "fixture-paracetamol-500",
        "Paracetamol",
        "Calpol 500",
        "GSK",
        "500 mg",
        "tablet",
        0.50,
        2.20,
        "Severe liver disease; hypersensitivity.",
        "https://janaushadhi.gov.in/",
        date(2026, 1, 1),
    ),
    MedicineFixture(
        "fixture-amlodipine-5",
        "Amlodipine Besylate",
        "Amlong 5",
        "Micro Labs",
        "5 mg",
        "tablet",
        1.00,
        3.50,
        "Hypersensitivity; use clinical guidance in severe hypotension.",
        "https://janaushadhi.gov.in/",
        date(2026, 1, 1),
    ),
    MedicineFixture(
        "fixture-metformin-500",
        "Metformin Hydrochloride",
        "Glycomet 500",
        "USV",
        "500 mg",
        "tablet",
        1.50,
        4.80,
        "Severe renal impairment; metabolic acidosis.",
        "https://janaushadhi.gov.in/",
        date(2026, 1, 1),
    ),
    MedicineFixture(
        "fixture-omeprazole-20",
        "Omeprazole",
        "Omez 20",
        "Dr. Reddy's",
        "20 mg",
        "capsule",
        2.00,
        8.00,
        "Hypersensitivity to proton pump inhibitors.",
        "https://janaushadhi.gov.in/",
        date(2026, 1, 1),
    ),
)


def build_rows(fixtures: tuple[MedicineFixture, ...] = FIXTURES) -> list[dict]:
    rows = []
    for fixture in fixtures:
        row = asdict(fixture)
        row["source_date"] = fixture.source_date.isoformat()
        row["canonical_text"] = canonicalize_medicine_text(
            generic_name=fixture.generic_name,
            brand_name=fixture.brand_name,
            manufacturer=fixture.manufacturer,
            strength=fixture.strength,
            dosage_form=fixture.dosage_form,
            contraindications=fixture.contraindications,
        )
        rows.append(row)
    return rows


FIELD_ALIASES = {
    "source_id": ("source_id", "id", "sku", "product_id", "set_id"),
    "drug_code": ("drug_code", "Drug Code"),
    "unit_size": ("unit_size", "Unit Size"),
    "therapeutic_group": ("therapeutic_group", "Group Name"),
    "generic_name": ("generic_name", "generic", "salt", "composition", "active_ingredient"),
    "brand_name": ("brand_name", "brand", "product_name", "medicine_name"),
    "manufacturer": ("manufacturer", "manufacturer_name", "company", "marketer"),
    "strength": ("strength", "dose", "dosage", "pack_strength"),
    "dosage_form": ("dosage_form", "form", "route", "pack_type"),
    "jan_aushadhi_mrp": ("jan_aushadhi_mrp", "jan_mrp", "pm", "generic_mrp"),
    "brand_mrp": ("brand_mrp", "mrp", "price", "maximum_retail_price"),
    "contraindications": (
        "contraindications",
        "contraindication",
        "warnings",
        "warnings_and_cautions",
    ),
    "source_url": ("source_url", "url", "source", "source_link"),
    "source_date": ("source_date", "updated_at", "last_updated", "effective_time"),
}


def _first_value(record: dict[str, Any], aliases: tuple[str, ...], default: Any = "") -> Any:
    for alias in aliases:
        value = record.get(alias)
        if value not in (None, "", []):
            return value
    return default


def normalize_records(records: list[dict[str, Any]], source_name: str) -> list[dict]:
    normalized = []
    for index, record in enumerate(records):
        values = {field: _first_value(record, aliases) for field, aliases in FIELD_ALIASES.items()}
        for field in (
            "generic_name",
            "brand_name",
            "manufacturer",
            "strength",
            "dosage_form",
            "contraindications",
        ):
            if isinstance(values[field], list):
                values[field] = "; ".join(str(item) for item in values[field])
        values["source_id"] = str(
            values["source_id"]
            or hashlib.sha256(
                f"{source_name}:{index}:{values['generic_name']}:{values['brand_name']}".encode()
            ).hexdigest()[:24]
        )
        if not values["generic_name"] or not values["brand_name"]:
            continue
        for field in ("jan_aushadhi_mrp", "brand_mrp"):
            try:
                values[field] = float(values[field]) if values[field] not in (None, "") else None
            except (TypeError, ValueError):
                values[field] = None
        values["source_url"] = str(values["source_url"] or source_name)
        values["source_date"] = str(values["source_date"] or date.today().isoformat())
        if len(values["source_date"]) == 8 and values["source_date"].isdigit():
            values["source_date"] = (
                f"{values['source_date'][:4]}-{values['source_date'][4:6]}-"
                f"{values['source_date'][6:]}"
            )
        values["canonical_text"] = canonicalize_medicine_text(
            generic_name=str(values["generic_name"]),
            brand_name=str(values["brand_name"]),
            manufacturer=str(values["manufacturer"]),
            strength=str(values["strength"]),
            dosage_form=str(values["dosage_form"]),
            contraindications=str(values["contraindications"]),
        )
        normalized.append(values)
    return normalized


def load_file(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8-sig") as file:
            return list(csv.DictReader(file))
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            return payload
        return payload.get("records", payload.get("data", []))
    raise ValueError("--input must be a .csv or .json file")


def load_jan_aushadhi_csv(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(newline="", encoding="utf-8-sig") as file:
        reader = csv.DictReader(file)
        required = {"Sr No", "Drug Code", "Generic Name", "Unit Size", "MRP", "Group Name"}
        if not required.issubset(reader.fieldnames or set()):
            raise ValueError(f"Jan Aushadhi CSV must contain columns: {sorted(required)}")
        for record in reader:
            generic_name = (record["Generic Name"] or "").strip()
            if not generic_name:
                continue
            unit_size = (record["Unit Size"] or "").strip()
            group_name = (record["Group Name"] or "").strip()
            rows.append(
                {
                    "source_id": f"jan-aushadhi-{record['Drug Code'].strip()}",
                    "drug_code": record["Drug Code"].strip(),
                    "unit_size": unit_size,
                    "therapeutic_group": group_name,
                    "generic_name": generic_name,
                    "brand_name": "Jan Aushadhi",
                    "manufacturer": "PMBI",
                    "strength": unit_size,
                    "dosage_form": group_name,
                    "jan_aushadhi_mrp": record["MRP"].strip(),
                    "brand_mrp": None,
                    "contraindications": "",
                    "source_url": str(path),
                    "source_date": date.today().isoformat(),
                }
            )
    return normalize_records(rows, str(path))


def load_openfda(query: str, max_records: int) -> list[dict[str, Any]]:
    response = httpx.get(
        "https://api.fda.gov/drug/label.json",
        params={"search": query, "limit": min(max_records, 1000)},
        timeout=30,
    )
    response.raise_for_status()
    records = []
    for record in response.json().get("results", []):
        openfda = record.get("openfda", {})
        records.append(
            {
                "source_id": _first_value(record, ("set_id", "id")),
                "generic_name": _first_value(openfda, ("generic_name",)),
                "brand_name": _first_value(openfda, ("brand_name",)),
                "manufacturer_name": _first_value(openfda, ("manufacturer_name",)),
                "strength": _first_value(record, ("active_ingredient",)),
                "dosage_form": _first_value(openfda, ("dosage_form", "route")),
                "contraindications": " ".join(
                    record.get("contraindications", []) + record.get("warnings", [])
                ),
                "source_url": "https://api.fda.gov/drug/label.json",
                "source_date": _first_value(record, ("effective_time",)),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed and upsert deterministic medicine fixtures."
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Embed and validate without uploading."
    )
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument(
        "--input", type=Path, help="CSV or JSON records to ingest instead of fixtures."
    )
    parser.add_argument("--jan-aushadhi-csv", type=Path, help="Official Jan Aushadhi catalog CSV.")
    parser.add_argument("--openfda-query", help="OpenFDA label search expression to import.")
    parser.add_argument("--max-records", type=int, default=100)
    parser.add_argument("--preview-rows", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.preview_rows < 0:
        raise SystemExit("--preview-rows cannot be negative")

    from fastembed import TextEmbedding

    started = time.perf_counter()
    if (
        sum(value is not None for value in (args.input, args.jan_aushadhi_csv, args.openfda_query))
        > 1
    ):
        raise SystemExit("choose one of --input, --jan-aushadhi-csv, or --openfda-query")
    if args.input:
        rows = normalize_records(load_file(args.input), str(args.input))
    elif args.jan_aushadhi_csv:
        rows = load_jan_aushadhi_csv(args.jan_aushadhi_csv)
    elif args.openfda_query:
        if args.max_records < 1:
            raise SystemExit("--max-records must be positive")
        rows = normalize_records(load_openfda(args.openfda_query, args.max_records), "openfda")
    else:
        rows = build_rows()
    model = TextEmbedding(model_name=MODEL_NAME)
    embeddings = list(model.embed([row["canonical_text"] for row in rows]))
    payloads = []
    for row, embedding in zip(rows, embeddings, strict=True):
        values = embedding.tolist()
        validate_embedding(values)
        payloads.append({**row, "embedding": values, "embedding_model": MODEL_NAME})

    if not args.dry_run:
        from app.db.client import get_supabase_client

        client = get_supabase_client()
        for offset in range(0, len(payloads), args.batch_size):
            client.table("medicines").upsert(
                payloads[offset : offset + args.batch_size], on_conflict="source_id"
            ).execute()

    elapsed = time.perf_counter() - started
    mode = "dry-run" if args.dry_run else "uploaded"
    print(f"{mode}: {len(payloads)} records, model={MODEL_NAME}, elapsed={elapsed:.2f}s")
    for payload in payloads[: args.preview_rows]:
        print(
            f"preview: {payload['source_id']} | {payload['generic_name']} | "
            f"unit/strength={payload['strength']} | mrp={payload['jan_aushadhi_mrp']} | "
            f"group/form={payload['dosage_form']}"
        )


if __name__ == "__main__":
    main()
