from __future__ import annotations

import argparse
import hashlib
import time
from pathlib import Path

import pdfplumber

from app.services.rag.canonicalize import validate_embedding

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def chunk_text(text: str, chunk_size: int = 1200, overlap: int = 150) -> list[str]:
    words = text.split()
    if chunk_size < 1 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap must be smaller than chunk_size")
    chunks = []
    step = chunk_size - overlap
    for start in range(0, len(words), step):
        chunk = " ".join(words[start : start + chunk_size]).strip()
        if chunk:
            chunks.append(chunk)
        if start + chunk_size >= len(words):
            break
    return chunks


def _clean_cell(value: str | None) -> str:
    return " ".join((value or "").replace("\n", " ").split())


def _join_cells(row: list[str | None], indexes: range) -> str:
    return _clean_cell(" ".join(row[index] or "" for index in indexes if index < len(row)))


def _pdf_chunk(
    source_id: str, pdf_path: Path, page_number: int, chunk_index: int, content: str
) -> dict:
    return {
        "source_id": source_id,
        "source_path": str(pdf_path),
        "page_number": page_number,
        "chunk_index": chunk_index,
        "content": content,
        "source_url": f"file://{pdf_path.resolve()}",
    }


def extract_cdsco_rows(pdf_path: Path) -> list[dict]:
    rows = []
    current: dict | None = None
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            for table in page.extract_tables():
                for row in table:
                    serial = _clean_cell(row[1] if len(row) > 1 else None).rstrip(".")
                    drug_name = _join_cells(row, range(4, 7))
                    notification = _join_cells(row, range(7, len(row)))
                    if serial.isdigit():
                        if current:
                            rows.append(current)
                        current = {
                            "serial": serial,
                            "drug_name": drug_name,
                            "notification": notification,
                            "page_number": page_number,
                        }
                    elif current:
                        if drug_name:
                            current["drug_name"] += f" {drug_name}"
                        if notification:
                            current["notification"] += f" {notification}"
    if current:
        rows.append(current)
    return [
        _pdf_chunk(
            f"cdsco-banned-{row['serial']}",
            pdf_path,
            row["page_number"],
            int(row["serial"]),
            f"Banned Drug Entry #{row['serial']}: "
            f"{row['drug_name'] or 'Unavailable in extracted table'} | "
            "Prohibited under Gazette Notification: "
            f"{row['notification'] or 'Unavailable in extracted table'}",
        )
        for row in rows
    ]


def extract_kendra_rows(pdf_path: Path) -> list[dict]:
    rows = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_number, page in enumerate(pdf.pages, 1):
            for table in page.extract_tables():
                for row in table:
                    if len(row) < 7 or not _clean_cell(row[0]).isdigit():
                        continue
                    code = _clean_cell(row[1])
                    content = (
                        f"Jan Aushadhi Kendra {code}: {_clean_cell(row[2])} | "
                        f"District: {_clean_cell(row[4])} | "
                        f"Pincode: {_clean_cell(row[5])} | "
                        f"Address: {_clean_cell(row[6])}"
                    )
                    rows.append(
                        _pdf_chunk(
                            f"kendra-karnataka-{code}",
                            pdf_path,
                            page_number,
                            int(_clean_cell(row[0])),
                            content,
                        )
                    )
    return rows


def extract_pdf_chunks(input_dir: Path, chunk_size: int, overlap: int) -> list[dict]:
    chunks = []
    for pdf_path in sorted(input_dir.glob("*.pdf")):
        if pdf_path.name == "cdsco_banned_drugs.pdf":
            chunks.extend(extract_cdsco_rows(pdf_path))
        elif pdf_path.name == "kendra_karnataka.pdf":
            chunks.extend(extract_kendra_rows(pdf_path))
        else:
            from pypdf import PdfReader

            reader = PdfReader(str(pdf_path))
            for page_number, page in enumerate(reader.pages, 1):
                text = page.extract_text() or ""
                for chunk_index, content in enumerate(chunk_text(text, chunk_size, overlap)):
                    source_id = hashlib.sha256(
                        f"{pdf_path.resolve()}:{page_number}:{chunk_index}".encode()
                    ).hexdigest()
                    chunks.append(
                        _pdf_chunk(
                            f"pdf-{source_id[:32]}", pdf_path, page_number, chunk_index, content
                        )
                    )
    return chunks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed and upsert downloaded medicine PDFs.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--batch-size", type=int, default=50)
    parser.add_argument("--chunk-size", type=int, default=1200)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument("--preview-rows", type=int, default=3)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input_dir.is_dir():
        raise SystemExit(f"PDF directory does not exist: {args.input_dir}")
    if args.batch_size < 1:
        raise SystemExit("--batch-size must be positive")
    if args.preview_rows < 0:
        raise SystemExit("--preview-rows cannot be negative")

    from fastembed import TextEmbedding

    started = time.perf_counter()
    chunks = extract_pdf_chunks(args.input_dir, args.chunk_size, args.overlap)
    if not chunks:
        raise SystemExit(f"No PDF text chunks found in {args.input_dir}")
    model = TextEmbedding(model_name=MODEL_NAME)
    payloads = []
    for chunk, embedding in zip(
        chunks, model.embed([item["content"] for item in chunks]), strict=True
    ):
        values = embedding.tolist()
        validate_embedding(values)
        payloads.append({**chunk, "embedding": values, "embedding_model": MODEL_NAME})

    if not args.dry_run:
        from app.db.client import get_supabase_client

        client = get_supabase_client()
        for offset in range(0, len(payloads), args.batch_size):
            client.table("medicine_chunks").upsert(
                payloads[offset : offset + args.batch_size], on_conflict="source_id"
            ).execute()

    mode = "dry-run" if args.dry_run else "uploaded"
    elapsed = time.perf_counter() - started
    print(f"{mode}: {len(payloads)} PDF chunks, model={MODEL_NAME}, elapsed={elapsed:.2f}s")
    by_source: dict[str, list[dict]] = {}
    for payload in payloads:
        by_source.setdefault(Path(payload["source_path"]).name, []).append(payload)
    for source_name, source_payloads in by_source.items():
        print(f"source: {source_name}, rows={len(source_payloads)}")
        for payload in source_payloads[: args.preview_rows]:
            print(f"preview: {payload['content']}")
        if len(source_payloads) > args.preview_rows:
            print(f"last: {source_payloads[-1]['content']}")


if __name__ == "__main__":
    main()
