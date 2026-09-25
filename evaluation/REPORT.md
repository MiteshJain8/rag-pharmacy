# Source lookup evaluation — 2026-09-25

## Corpus and method

The live Supabase audit found 3,443 `medicines` records and 2,204 `medicine_chunks` records, all 5,647 with embeddings. The 60 fixed questions in `cases.jsonl` comprise 45 exact-record lookups (15 Jan Aushadhi, 10 OpenFDA, 10 CDSCO, 10 Kendra), 10 personal medical advice requests, and 5 unrelated requests. Positive target IDs were checked against sampled live records; the questions are source-derived, not independent user queries. The same file and database were used for both runs. Neither run wrote to Supabase.

A read-only checksum of `source_id`, source text, and `updated_at` stayed the same before and after both runs: `medicines` 3,443 rows / `65861995a6b5534c19e4d0fe31a1c2a0`; `medicine_chunks` 2,204 rows / `ff72c5cd2407f1cea7b4e560c02164b5`. This checks the tested content, but is not an immutable database snapshot of every field.

## Results

- **Expected record in top 5:** baseline 44/45 (97.8%); improved 45/45 (100%).
- **Mean reciprocal rank at 5:** baseline 0.8167; improved 0.9019.
- **Withheld source cards for unsafe or unrelated requests:** baseline 0/15; improved 15/15. The unsafe subset improved from 0/10 to 10/10; unrelated from 0/5 to 5/5.
- **Every returned source ID present in the answer:** baseline 0%; improved 100%. This checks ID citation structure, not whether the underlying source is correct or current.
- **Full displayed source extract and ID copied into the answer:** baseline 0%; improved 100%. This is a strict mechanical support check, not a medical fact check.
- **Warm end-to-end query latency:** baseline p50 2,525.5 ms and p95 4,181.6 ms; improved p50 1,225.5 ms and p95 1,942.2 ms. The first case is excluded from each latency distribution. Requests were sequential and include network time.
- **Provider calls:** baseline attempted 60 Groq completions and 60 Cohere reranks; improved attempted 0 Groq completions and 45 Cohere reranks. The improved provider response reported 12 Cohere search units in 12 cases. The older provider client did not record billing units or Groq tokens, so provider cost cannot be compared reliably. Provider calls are attempts, not confirmed charges.

Each run initially had transient `Server disconnected` errors. The benchmark resumed only those failed cases and completed all 60. A one-retry transport guard was added to repository reads after the measured run. It does not change successful ranking behavior, but its effect on production error rate has not been measured.

## Interpretation and limits

These numbers show lookup behavior on a small, source-derived set. The 100% improved recall is a **45-question result**, not a general retrieval guarantee. Queries containing an exact Kendra code receive a direct record lookup; CDSCO queries use focused term search; these are appropriate for the evaluation's lookup intent but make it easier than open-ended medical search. The abstention check covers only 15 examples, and the rule can reject some valid semantic-only queries. The content includes imported catalog values and extracted PDFs; no pharmacist or clinician reviewed source truth, staleness, prices, dosage, or regulatory status. Source IDs are corpus record IDs rather than independently verified primary-source citations.

For deployment, Groq and Cohere keys are intentionally absent from `render.yaml`, so the public demo uses deterministic extracts with no provider calls. Performance on Render's free instance, including cold starts, remains unmeasured until deployment.
