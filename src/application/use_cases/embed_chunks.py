"""
Use-case: Nhúng các chunk chưa có vector.
Được thiết kế dạng Class với Dependency Injection (Clean Architecture).

Mỗi batch được ghi SONG SONG vào SQLite (chunk_embedding.vector_data) và Qdrant.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, List, Optional, Tuple

from src.domain.models import VectorRecord
from src.domain.ports import EmbeddingPort, SqlDatabasePort, VectorDatabasePort


@dataclass
class EmbedStats:
    """Thống kê quá trình nhúng.

    total: số chunk cần xử lý (chưa nhúng).
    successful: số chunk đã nhúng và lưu vào SQLite.
    failed: số chunk nhúng thất bại (sau hết retry).
    failed_ids: chunk_id nhúng thất bại.
    qdrant_failed_ids: chunk_id đã lưu SQLite nhưng đẩy Qdrant lỗi. Các chunk này đã có vector trong
        SQLite nên lần chạy sau sẽ không nhúng lại; cần đồng bộ lên Qdrant riêng.
    """

    total: int = 0
    successful: int = 0
    failed: int = 0
    failed_ids: List[str] = field(default_factory=list)
    qdrant_failed_ids: List[str] = field(default_factory=list)


class EmbedChunksUseCase:
    """Use case nhúng các chunk văn bản chưa có vector và đồng bộ vào DB/VectorDB."""

    def __init__(
        self,
        sql_db: SqlDatabasePort,
        embedding_service: EmbeddingPort,
        vector_db: Optional[VectorDatabasePort] = None,
    ) -> None:
        self._sql_db = sql_db
        self._embedding_service = embedding_service
        self._vector_db = vector_db

    def execute(
        self,
        provider: str,
        model_name: Optional[str] = None,
        batch_size: int = 16,
        limit: Optional[int] = None,
        dimension: Optional[int] = None,
        max_retries: int = 3,
        push_qdrant: bool = False,
        qdrant_collection: str = "chunks",
    ) -> EmbedStats:
        """Thực thi tiến trình nhúng các chunk chưa có vector."""
        if push_qdrant and self._vector_db is None:
            raise ValueError("push_qdrant=True nhưng use case chưa được tiêm vector_db.")

        model_name_resolved, dim = self._resolve_dimension(dimension)
        if model_name:
            model_name_resolved = model_name

        if push_qdrant and self._vector_db:
            self._prepare_collection(qdrant_collection, dim)

        pending = self._sql_db.fetch_unembedded_chunks(
            provider=provider,
            model_name=model_name_resolved,
            dimension=dim,
            limit=limit,
        )
        stats = EmbedStats(total=len(pending))
        done = 0

        with ThreadPoolExecutor(max_workers=2) as pool:
            for batch in self._chunked(pending, batch_size):
                chunk_ids, texts = zip(*batch)

                vectors = self._embed_batch_with_retry(list(texts), max_retries)
                if not vectors or len(vectors) != len(texts):
                    stats.failed += len(batch)
                    stats.failed_ids.extend(chunk_ids)
                    continue

                self._save_batch_parallel(
                    pool=pool,
                    batch_chunk_ids=chunk_ids,
                    batch_texts=texts,
                    vectors=vectors,
                    provider=provider,
                    model_name=model_name_resolved,
                    dimension=dim,
                    push_qdrant=push_qdrant,
                    qdrant_collection=qdrant_collection,
                    stats=stats,
                )

                done += len(batch)
                print(f"[OK] Batch {done}/{stats.total}")

        return stats

    def _resolve_dimension(self, override: Optional[int]) -> Tuple[str, int]:
        """Nếu có override dimension thì dùng ngay; ngược lại nhúng câu mẫu để lấy dimension."""
        if override is not None:
            return "", override
        test_vec = self._embedding_service.embed_text("probe")
        return test_vec.model_name, len(test_vec.values)

    def _prepare_collection(self, collection: str, dim: int) -> None:
        """Tạo collection nếu chưa có; nếu đã có thì kiểm tra số chiều khớp."""
        assert self._vector_db is not None
        if not self._vector_db.collection_exists(collection):
            self._vector_db.create_collection(collection, dim)
            return
        existing = self._vector_db.get_collection_dimension(collection)
        if existing is not None and existing != dim:
            raise ValueError(
                f"Collection '{collection}' có {existing} chiều nhưng model sinh vector {dim} chiều. "
                "Hãy dùng collection khác hoặc đổi model cho khớp."
            )

    def _embed_batch_with_retry(self, texts: List[str], max_retries: int) -> List[Any]:
        """Thực hiện nhúng batch kèm retry cơ chế exponential backoff."""
        attempt = 0
        while attempt <= max_retries:
            try:
                return self._embedding_service.embed_batch(texts)
            except Exception as exc:  # noqa: BLE001
                attempt += 1
                if attempt > max_retries:
                    break
                backoff = 2 ** attempt
                print(f"[WARN] Retry {attempt}/{max_retries} sau {backoff}s vi: {exc}")
                time.sleep(backoff)
        return []

    def _save_batch_parallel(
        self,
        pool: ThreadPoolExecutor,
        batch_chunk_ids: Tuple[str, ...],
        batch_texts: Tuple[str, ...],
        vectors: List[Any],
        provider: str,
        model_name: str,
        dimension: int,
        push_qdrant: bool,
        qdrant_collection: str,
        stats: EmbedStats,
    ) -> None:
        """Ghi đồng thời vào SQLite và Qdrant qua thread pool."""
        records = [
            {
                "chunk_id": cid,
                "provider": provider,
                "model_name": model_name,
                "dimension": dimension,
                "vector": vec.values,
                "token_count": vec.token_count,
            }
            for cid, vec in zip(batch_chunk_ids, vectors)
        ]
        sql_future = pool.submit(self._sql_db.save_embeddings_batch, records)

        qdrant_future = None
        if push_qdrant and self._vector_db:
            points = [
                VectorRecord(
                    id=cid,
                    vector=vec.values,
                    payload={
                        "chunk_id": cid,
                        "text": text,
                        "provider": provider,
                        "model_name": model_name,
                        "dimension": dimension,
                        "token_count": vec.token_count,
                    },
                )
                for cid, text, vec in zip(batch_chunk_ids, batch_texts, vectors)
            ]
            qdrant_future = pool.submit(self._vector_db.upsert, qdrant_collection, points)

        # Chờ SQLite: lỗi ở đây là lỗi nghiêm trọng
        sql_future.result()
        stats.successful += len(batch_chunk_ids)

        if qdrant_future is not None:
            try:
                qdrant_future.result()
            except Exception as exc:  # noqa: BLE001
                stats.qdrant_failed_ids.extend(batch_chunk_ids)
                print(f"[WARN] Qdrant loi o batch {len(batch_chunk_ids)} chunk (SQLite van da luu): {exc}")

    @staticmethod
    def _chunked(seq: List[Tuple[str, str]], size: int):
        for i in range(0, len(seq), size):
            yield seq[i: i + size]

