import re
import time
from typing import Any

import httpx
from supabase import Client


class MedicineRepository:
    def __init__(self, client: Client) -> None:
        self.client = client

    @staticmethod
    def _execute_read(request: Any) -> Any:
        for attempt in range(2):
            try:
                return request.execute()
            except httpx.TransportError:
                if attempt:
                    raise
                time.sleep(0.2)
        raise RuntimeError("Read retry exhausted")

    def dense_search(self, embedding: list[float], limit: int = 25) -> list[dict[str, Any]]:
        medicine_response = self.client.rpc(
            "match_medicines_dense", {"query_embedding": embedding, "match_count": limit}
        )
        medicine_response = self._execute_read(medicine_response)
        chunk_response = self.client.rpc(
            "match_medicine_chunks_dense", {"query_embedding": embedding, "match_count": limit}
        )
        chunk_response = self._execute_read(chunk_response)
        return self._merge(
            medicine_response.data or [], chunk_response.data or [], "similarity", limit
        )

    def sparse_search(self, query: str, limit: int = 25) -> list[dict[str, Any]]:
        medicine_response = self.client.rpc(
            "match_medicines_sparse", {"query_text": query, "match_count": limit}
        )
        medicine_response = self._execute_read(medicine_response)
        chunk_response = self.client.rpc(
            "match_medicine_chunks_sparse", {"query_text": query, "match_count": limit}
        )
        chunk_response = self._execute_read(chunk_response)
        return self._merge(
            medicine_response.data or [], chunk_response.data or [], "lexical_score", limit
        )

    def lookup_chunk(self, source_id: str) -> dict[str, Any] | None:
        request = (
            self.client.table("medicine_chunks")
            .select("source_id,content,source_path,source_url,page_number,chunk_index")
            .eq("source_id", source_id)
            .limit(1)
        )
        response = self._execute_read(request)
        if not response.data:
            return None
        row = response.data[0]
        row["canonical_text"] = row.pop("content")
        row["lexical_score"] = 1.0
        return row

    def search_cdsco_terms(self, query: str) -> list[dict[str, Any]]:
        terms = [
            token
            for token in re.findall(r"[a-z0-9]+", query.lower())
            if len(token) >= 5
            and token not in {"cdsco", "banned", "notification", "combination", "systemic"}
        ]
        if not terms:
            return []
        request = (
            self.client.table("medicine_chunks")
            .select("source_id,content,source_path,source_url,page_number,chunk_index")
            .like("source_id", "cdsco-%")
            .ilike("content", f"%{terms[0]}%")
            .limit(100)
        )
        response = self._execute_read(request)
        rows = response.data or []
        for row in rows:
            text = row["content"].lower()
            row["lexical_score"] = sum(token in text for token in terms) / len(terms)
            row["canonical_text"] = row.pop("content")
        return sorted(rows, key=lambda row: row["lexical_score"], reverse=True)[:25]

    @staticmethod
    def _merge(
        medicines: list[dict[str, Any]],
        chunks: list[dict[str, Any]],
        score_key: str,
        limit: int,
    ) -> list[dict[str, Any]]:
        records = {item["source_id"]: item for item in medicines}
        records.update({item["source_id"]: item for item in chunks})
        return sorted(records.values(), key=lambda item: item.get(score_key, 0), reverse=True)[
            :limit
        ]
