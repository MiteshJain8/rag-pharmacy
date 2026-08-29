from typing import Any

from supabase import Client


class MedicineRepository:
    def __init__(self, client: Client) -> None:
        self.client = client

    def dense_search(self, embedding: list[float], limit: int = 25) -> list[dict[str, Any]]:
        medicine_response = self.client.rpc(
            "match_medicines_dense", {"query_embedding": embedding, "match_count": limit}
        ).execute()
        chunk_response = self.client.rpc(
            "match_medicine_chunks_dense", {"query_embedding": embedding, "match_count": limit}
        ).execute()
        return self._merge(
            medicine_response.data or [], chunk_response.data or [], "similarity", limit
        )

    def sparse_search(self, query: str, limit: int = 25) -> list[dict[str, Any]]:
        medicine_response = self.client.rpc(
            "match_medicines_sparse", {"query_text": query, "match_count": limit}
        ).execute()
        chunk_response = self.client.rpc(
            "match_medicine_chunks_sparse", {"query_text": query, "match_count": limit}
        ).execute()
        return self._merge(
            medicine_response.data or [], chunk_response.data or [], "lexical_score", limit
        )

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
