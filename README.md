# Indian Pharma & Generic Drug Substitute Intelligence Engine

Standalone FastAPI foundation for hybrid retrieval over Indian medicine and generic-substitution data.

## Local Setup

Requirements: Python 3.11+ and `uv`.

```bash
uv sync
cp .env.example .env
uv run pytest
uv run ruff check .
uv run uvicorn app.main:app --reload
```

The service serves a basic browser UI at `/`, plus `GET /health` and `POST /api/v1/query`. The UI is static HTML served by FastAPI, so it deploys with the same Python container and requires no Node.js build step.

## Supabase Setup

1. Create or select a Supabase project with the `vector` extension available.
2. Apply [supabase_schema.sql](supabase_schema.sql) in the SQL editor or through the project's migration workflow.
3. Set `SUPABASE_URL` and `SUPABASE_SERVICE_ROLE_KEY` in `.env`. Keep the service-role key server-side.
4. Run a dry ingestion first:

```bash
uv run python scripts/ingest_data.py --dry-run
uv run python scripts/ingest_data.py --batch-size 50
```

The first ingestion run downloads the FastEmbed model. The script upserts deterministic fixtures by `source_id`.

For larger open datasets, export PMBI/Jan Aushadhi or a reviewed Kaggle CSV/JSON and import it through the normalized loader. Common aliases such as `salt`, `brand`, `company`, `dose`, `form`, and `mrp` are accepted:

```bash
uv run python scripts/ingest_data.py --input ./data/medicines.csv --batch-size 100
```

For the provided catalog, use the explicit mapping and validate before upload:

```bash
uv run python scripts/ingest_data.py \
	--jan-aushadhi-csv ./data/jan_aushadhi_products.csv \
	--dry-run --preview-rows 3
```

OpenFDA clinical labels can be imported separately. They provide clinical text and provenance, but generally do not provide Indian MRPs, so pricing remains unavailable for those records:

```bash
uv run python scripts/ingest_data.py \
	--openfda-query 'openfda.product_type:"HUMAN OTC DRUG"' \
	--max-records 1000 --batch-size 100
```

Review and deduplicate external records before production import. Do not treat OpenFDA, community datasets, or PMBI catalog values as interchangeable clinical advice; retain source dates and URLs.

For manually downloaded catalogs and official PDFs, place files in a local directory and run the chunk importer:

```bash
uv run python scripts/ingest_pdfs.py --input-dir ./data/pdfs --dry-run
uv run python scripts/ingest_pdfs.py --input-dir ./data/pdfs --batch-size 100
```

The two supplied grid PDFs are extracted row-by-row with `pdfplumber`: CDSCO continuation lines are coalesced into one banned-drug entry, and each Karnataka Kendra row keeps its code, name, district, pincode, and address together. Other PDFs use the generic overlapping text fallback. Dry runs print per-file counts plus first/last samples. Apply the updated `supabase_schema.sql` before the first PDF upload.

To enable enhanced query rewriting and answer synthesis, add `GROQ_API_KEY` and `GROQ_MODEL` to `.env`. To enable cross-encoder reranking, add `COHERE_API_KEY` and `COHERE_RERANK_MODEL`. These are optional; the service remains usable with grounded deterministic responses when they are absent or temporarily unavailable.

## Retrieval Design

Dense pgvector cosine search and PostgreSQL lexical search each return up to 25 records. Reciprocal Rank Fusion with `k=60` creates a pool of up to 50 records, which is intended for reranking to the best 5 evidence chunks. The initial lexical ranker is `ts_rank_cd`; native PostgreSQL `tsvector` is not BM25.

## Free-Tier Deployment Notes

Run the API on a small Render or DigitalOcean container and use Supabase's transaction pooler for database access. Configure provider keys only as server-side environment variables. Before public deployment, add rate limiting and source/clinical review for medical content and price freshness.
