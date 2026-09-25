"""Reproducible fixed-query evaluation. Run from the repository root."""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

from app.services.rag.service import RagService

CASES = Path(__file__).with_name("cases.jsonl")


def percentile(values: list[float], percent: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    index = (len(ordered) - 1) * percent
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    return round(ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower), 1)


def answer_supported(answer: str, source: dict) -> bool:
    """Mechanical check that the answer's displayed fact comes from its cited record."""
    source_id = source["source_id"]
    if f"[{source_id}]" not in answer:
        return False
    content = source["content"]
    if source_id.startswith("cdsco-"):
        return content.split("|", 1)[0].removeprefix("Banned Drug Entry ").strip() in answer
    if source_id.startswith(("jan-aushadhi-", "kendra-")):
        return content in answer
    metadata = source.get("metadata", {})
    name = metadata.get("brand_name") or metadata.get("generic_name") or "This product"
    return str(name) in answer


async def evaluate(mode: str, limit: int, expand_query: bool, output: Path, resume: bool) -> None:
    cases = [json.loads(line) for line in CASES.read_text().splitlines() if line.strip()]
    if len(cases) != 60:
        raise ValueError("Expected exactly 60 fixed cases")
    cases = cases[:limit]
    previous = {}
    if resume and output.exists():
        previous = {item["case"]: item for item in json.loads(output.read_text())["results"]}
    results = []
    for number, case in enumerate(cases, 1):
        if number in previous and not previous[number].get("error"):
            results.append(previous[number])
            continue
        try:
            service = RagService()
            calls = {"groq": 0, "cohere": 0}
            original_complete = service.groq.complete
            original_rerank = service.cohere.rerank

            async def counted_complete(prompt: str) -> str | None:
                calls["groq"] += 1
                return await original_complete(prompt)

            async def counted_rerank(query: str, documents: list[str], top_n: int):
                if documents and service.cohere.api_key:
                    calls["cohere"] += 1
                return await original_rerank(query, documents, top_n)

            service.groq.complete = counted_complete
            service.cohere.rerank = counted_rerank
            started = time.perf_counter()
            result = await service.query(case["query"], 5, expand_query)
            elapsed_ms = round((time.perf_counter() - started) * 1000, 1)
            ids = [source["source_id"] for source in result.sources]
            expected = case.get("expected_source_id")
            answer = result.answer
            citation_complete = len(ids) == 1 and answer.count(f"[{ids[0]}]") == 1 if ids else None
            support = answer_supported(answer, result.sources[0]) if len(ids) == 1 else None
            abstained = not ids and ("cannot" in answer.lower() or "no matching" in answer.lower())
            results.append(
                {
                    "case": number,
                    "category": case["category"],
                    "query": case["query"],
                    "expected_source_id": expected,
                    "selected_ids": ids,
                    "exact_selected_source": ids == [expected] if expected else None,
                    "single_source": len(ids) == 1 if expected else None,
                    "abstained": abstained if case.get("expected_refusal") else None,
                    "citation_complete": citation_complete if ids else None,
                    "answer_supported": support,
                    "answer": answer,
                    "latency_ms": elapsed_ms,
                    "provider_usage": result.retrieval.get("provider_usage", {}),
                    "provider_calls": calls,
                    "error": None,
                }
            )
        except Exception as error:
            results.append({"case": number, "category": case["category"], "error": str(error)})
        status = results[-1].get("error") or "done"
        print(f"{number}/{len(cases)} {case['category']}: {status}", flush=True)
    positive = [r for r in results if r.get("exact_selected_source") is not None]
    negative = [r for r in results if r.get("abstained") is not None]
    cited = [r for r in results if r.get("citation_complete") is not None]
    supported = [r for r in results if r.get("answer_supported") is not None]
    latencies = [r["latency_ms"] for r in results if isinstance(r.get("latency_ms"), (int, float))]
    summary = {
        "mode": mode,
        "date_utc": datetime.now(timezone.utc).isoformat(),
        "cases_attempted": len(cases),
        "cases_completed": len(results) - sum(bool(r.get("error")) for r in results),
        "positive_cases": len(positive),
        "negative_cases": len(negative),
        "exact_selected_source_rate": round(
            statistics.mean(r["exact_selected_source"] for r in positive), 4
        )
        if positive
        else None,
        "single_source_rate": round(statistics.mean(r["single_source"] for r in positive), 4)
        if positive
        else None,
        "abstention_rate": round(statistics.mean(r["abstained"] for r in negative), 4)
        if negative
        else None,
        "citation_id_coverage": round(statistics.mean(r["citation_complete"] for r in cited), 4)
        if cited
        else None,
        "answer_support_rate": round(
            statistics.mean(r["answer_supported"] for r in supported), 4
        )
        if supported
        else None,
        "provider_calls": {
            name: sum(r.get("provider_calls", {}).get(name, 0) for r in results)
            for name in ("groq", "cohere")
        },
        "provider_reported_usage": {
            "groq_total_tokens": sum(
                r.get("provider_usage", {}).get("groq", {}).get("total_tokens", 0) for r in results
            )
            if any(r.get("provider_usage", {}).get("groq") for r in results)
            else None,
            "cohere_search_units": sum(
                r.get("provider_usage", {}).get("cohere", {}).get("search_units", 0)
                for r in results
            )
            if any(r.get("provider_usage", {}).get("cohere") for r in results)
            else None,
        },
        "warm_latency_p50_ms": percentile(latencies[1:], 0.50),
        "warm_latency_p95_ms": percentile(latencies[1:], 0.95),
        "limitations": (
            "Fixed, source-derived queries; exact selected IDs measure lookup quality only. "
            "Answer support checks a cited phrase from the selected record; it does not "
            "prove medical correctness or current validity. First request excluded "
            "from warm latency. Provider charges require current account pricing."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"summary": summary, "results": results}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["baseline", "improved"])
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--expand-query", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(evaluate(args.mode, args.limit, args.expand_query, args.output, args.resume))
