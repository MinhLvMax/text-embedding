import array
import threading
from pathlib import Path
import json
from typing import Optional, Dict, Any, List, Tuple
from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker
from src.domain.ports.sql_database_port import SqlDatabasePort
from src.infrastructure.sql_database_adapters.models import Base, Chunk, ChunkEmbedding

class ChunkRepository:
    # KHÓA CẤP CLASS (Class-level lock):
    # Dùng chung cho TOÀN BỘ các instance của ChunkRepository trong tiến trình.
    # Ngăn tuyệt đối tình trạng 2 worker độc lập khởi tạo 2 instance repo khác nhau
    # rồi bốc trùng 1 chunk.
    _lock = threading.Lock()

    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    def fetch_pending(self, limit: Optional[int] = None) -> List[Tuple[str, str]]:
        """Trả về [(chunk_id, text gửi cho LLM)] của các chunk PENDING."""
        with self._lock:
            with self._session_factory() as session:
                query = (
                    session.query(Chunk)
                    .filter(Chunk.status == "PENDING")
                    .order_by(Chunk.chunk_id)
                )
                if limit:
                    query = query.limit(limit)
                return [
                    (c.chunk_id, c.chunk_overlap_200 or c.text)
                    for c in query.all()
                ]

    def claim_next_pending(self) -> Optional[Tuple[str, str]]:
        """Lấy nguyên tử (atomic) đúng 1 chunk PENDING và chuyển ngay sang PROCESSING.

        Sử dụng kết hợp 2 lớp bảo vệ chống Race Condition:
        1. Class-level threading.Lock: Đồng bộ hóa toàn bộ các thread trong cùng tiến trình.
        2. SQLite UPDATE ... RETURNING nguyên tử: Đảm bảo độc quyền ngay cả khi chạy
           trên nhiều tiến trình / terminal khác nhau.
        Trả về (chunk_id, text) hoặc None nếu không còn chunk nào PENDING.
        """
        with self._lock:
            with self._session_factory() as session:
                stmt = text(
                    """
                    UPDATE chunk
                    SET status = 'PROCESSING'
                    WHERE chunk_id = (
                        SELECT chunk_id FROM chunk
                        WHERE status = 'PENDING'
                        ORDER BY chunk_id
                        LIMIT 1
                    )
                    RETURNING chunk_id, text, chunk_overlap_200
                    """
                )
                row = session.execute(stmt).fetchone()
                session.commit()
                if not row:
                    return None
                chunk_id, text_val, overlap_val = row
                return (chunk_id, overlap_val or text_val)

    def reset_processing_to_pending(self) -> int:
        """Khôi phục các chunk đang bị kẹt ở trạng thái PROCESSING về lại PENDING.

        Rất hữu ích khi một tiến trình worker bị dừng đột ngột (mất điện, tắt console, Ctrl+C).
        Trả về số lượng chunk đã được khôi phục.
        """
        with self._lock:
            with self._session_factory() as session:
                count = (
                    session.query(Chunk)
                    .filter(Chunk.status == "PROCESSING")
                    .update({"status": "PENDING"})
                )
                session.commit()
                return count

    def update_provider(self, chunk_id: str, provider: str) -> None:
        """Cập nhật tên nhà cung cấp (provider) thực hiện trích xuất cho chunk."""
        with self._lock:
            with self._session_factory() as session:
                chunk = session.get(Chunk, chunk_id)
                if chunk:
                    chunk.provider = provider
                    session.commit()

    def update_model(self, chunk_id: str, model: str) -> None:
        """Cập nhật tên model thực hiện trích xuất cho chunk."""
        with self._lock:
            with self._session_factory() as session:
                chunk = session.get(Chunk, chunk_id)
                if chunk:
                    chunk.model = model
                    session.commit()

    def mark_completed(
            self,
            chunk_id: str,
            result: Dict[str, Any],
            provider: Optional[str] = None,
            model: Optional[str] = None,
    ) -> None:
        """Đánh dấu chunk đã xử lý thành công, lưu kết quả trích xuất và tùy chọn cập nhật provider/model."""
        with self._lock:
            with self._session_factory() as session:
                chunk = session.get(Chunk, chunk_id)
                if chunk:
                    chunk.graph_extraction = result
                    chunk.status = "COMPLETED"
                    if provider:
                        chunk.provider = provider
                    if model:
                        chunk.model = model
                    session.commit()

    def mark_failed(self, chunk_id: str, max_retries: int) -> str:
        """Tăng retry_count; FAILED nếu vượt ngưỡng, ngược lại giữ PENDING. Trả về status mới."""
        with self._lock:
            with self._session_factory() as session:
                chunk = session.get(Chunk, chunk_id)
                if chunk:
                    chunk.retry_count += 1
                    chunk.status = "FAILED" if chunk.retry_count >= max_retries else "PENDING"
                    session.commit()
                    return chunk.status
                return "NOT_FOUND"


# ==============================================================================
# TIỆN ÍCH CHUYỂN ĐỔI VECTOR <-> BLOB (FLOAT32 NHỊ PHÂN)
# ==============================================================================

def vector_to_blob(vector: List[float]) -> bytes:
    """Chuyển danh sách float thành chuỗi byte float32 nhị phân (BLOB)."""
    return array.array("f", vector).tobytes()


def blob_to_vector(blob_data: bytes) -> List[float]:
    """Chuyển chuỗi byte BLOB ngược lại thành danh sách float32."""
    arr = array.array("f")
    arr.frombytes(blob_data)
    return arr.tolist()


# ==============================================================================
# REPOSITORY QUẢN LÝ VECTOR EMBEDDING (CHUNK_EMBEDDING)
# ==============================================================================

class ChunkEmbeddingRepository:
    """
    Repository thao tác với bảng chunk_embedding.
    Hỗ trợ:
    - Quét các chunk chưa được tạo vector cho model/dimension cụ thể.
    - Lưu / Cập nhật vector dạng BLOB (Idempotent UPSERT).
    - Xuất dữ liệu kèm payload để đẩy sang Qdrant.
    """

    _lock = threading.Lock()

    def __init__(self, session_factory: sessionmaker):
        self._session_factory = session_factory

    def fetch_unembedded_chunks(
        self,
        provider: str,
        model_name: str,
        dimension: int,
        limit: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        """
        Lấy danh sách các chunk (chunk_id, text) CHƯA có vector nhúng
        cho bộ ba (provider, model_name, dimension).
        Ưu tiên dùng chunk_overlap_200 nếu có, ngược lại dùng text.
        """
        with self._lock:
            with self._session_factory() as session:
                sql = """
                    SELECT c.chunk_id, COALESCE(c.chunk_overlap_200, c.text) AS content
                    FROM chunk c
                    LEFT JOIN chunk_embedding ce
                        ON c.chunk_id = ce.chunk_id
                        AND ce.provider = :provider
                        AND ce.model_name = :model_name
                        AND ce.dimension = :dimension
                    WHERE ce.id IS NULL
                    ORDER BY c.chunk_id
                """
                if limit:
                    sql += f" LIMIT {int(limit)}"

                stmt = text(sql)
                rows = session.execute(
                    stmt,
                    {
                        "provider": provider,
                        "model_name": model_name,
                        "dimension": dimension,
                    },
                ).fetchall()
                return [(r[0], r[1]) for r in rows]

    def save_embedding(
        self,
        chunk_id: str,
        provider: str,
        model_name: str,
        dimension: int,
        vector: List[float],
        token_count: Optional[int] = None,
    ) -> None:
        """Lưu hoặc cập nhật (UPSERT) vector nhúng cho 1 chunk."""
        blob = vector_to_blob(vector)
        with self._lock:
            with self._session_factory() as session:
                stmt = text(
                    """
                    INSERT INTO chunk_embedding (chunk_id, provider, model_name, dimension, token_count, vector_data)
                    VALUES (:chunk_id, :provider, :model_name, :dimension, :token_count, :vector_data)
                    ON CONFLICT(chunk_id, provider, model_name, dimension)
                    DO UPDATE SET
                        vector_data = excluded.vector_data,
                        token_count = excluded.token_count,
                        created_at = CURRENT_TIMESTAMP
                    """
                )
                session.execute(
                    stmt,
                    {
                        "chunk_id": chunk_id,
                        "provider": provider,
                        "model_name": model_name,
                        "dimension": dimension,
                        "token_count": token_count,
                        "vector_data": blob,
                    },
                )
                session.commit()

    def save_embeddings_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> int:
        """
        Lưu hàng loạt (batch UPSERT) danh sách vector nhúng.
        Mỗi record là dict gồm:
        {chunk_id, provider, model_name, dimension, vector, token_count}
        """
        if not records:
            return 0

        prepared_params = [
            {
                "chunk_id": r["chunk_id"],
                "provider": r["provider"],
                "model_name": r["model_name"],
                "dimension": r["dimension"],
                "token_count": r.get("token_count"),
                "vector_data": vector_to_blob(r["vector"]),
            }
            for r in records
        ]

        with self._lock:
            with self._session_factory() as session:
                stmt = text(
                    """
                    INSERT INTO chunk_embedding (chunk_id, provider, model_name, dimension, token_count, vector_data)
                    VALUES (:chunk_id, :provider, :model_name, :dimension, :token_count, :vector_data)
                    ON CONFLICT(chunk_id, provider, model_name, dimension)
                    DO UPDATE SET
                        vector_data = excluded.vector_data,
                        token_count = excluded.token_count,
                        created_at = CURRENT_TIMESTAMP
                    """
                )
                session.execute(stmt, prepared_params)
                session.commit()
                return len(prepared_params)

    def get_embedding(
        self,
        chunk_id: str,
        provider: str,
        model_name: str,
        dimension: int,
    ) -> Optional[Dict[str, Any]]:
        """Lấy vector nhúng và metadata đã giải mã (List[float]) của một chunk."""
        with self._lock:
            with self._session_factory() as session:
                stmt = text(
                    """
                    SELECT id, chunk_id, provider, model_name, dimension, token_count, vector_data, created_at
                    FROM chunk_embedding
                    WHERE chunk_id = :chunk_id
                      AND provider = :provider
                      AND model_name = :model_name
                      AND dimension = :dimension
                    LIMIT 1
                    """
                )
                row = session.execute(
                    stmt,
                    {
                        "chunk_id": chunk_id,
                        "provider": provider,
                        "model_name": model_name,
                        "dimension": dimension,
                    },
                ).fetchone()

                if not row:
                    return None

                return {
                    "id": row[0],
                    "chunk_id": row[1],
                    "provider": row[2],
                    "model_name": row[3],
                    "dimension": row[4],
                    "token_count": row[5],
                    "vector": blob_to_vector(row[6]),
                    "created_at": str(row[7]),
                }

    def export_for_qdrant(
        self,
        provider: str,
        model_name: str,
        dimension: int,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """
        Xuất danh sách vector kết hợp payload từ bảng chunk để đẩy sang Qdrant.
        Trả về format chuẩn:
        [
            {
                "id": chunk_id,
                "vector": [float, ...],
                "payload": {
                    "text": str,
                    "metadata": dict,
                    "token_count": int,
                    "provider": str,
                    "model_name": str
                }
            },
            ...
        ]
        """
        with self._lock:
            with self._session_factory() as session:
                sql = """
                    SELECT
                        ce.chunk_id,
                        ce.vector_data,
                        ce.token_count,
                        c.text,
                        c.chunk_overlap_200,
                        c.chunk_metadata
                    FROM chunk_embedding ce
                    INNER JOIN chunk c ON ce.chunk_id = c.chunk_id
                    WHERE ce.provider = :provider
                      AND ce.model_name = :model_name
                      AND ce.dimension = :dimension
                    ORDER BY ce.id
                """
                if limit:
                    sql += f" LIMIT {int(limit)} OFFSET {int(offset)}"

                rows = session.execute(
                    text(sql),
                    {
                        "provider": provider,
                        "model_name": model_name,
                        "dimension": dimension,
                    },
                ).fetchall()

                results: List[Dict[str, Any]] = []
                for r in rows:
                    chunk_id, blob_val, token_val, raw_text, overlap_val, meta_val = r
                    results.append({
                        "id": chunk_id,
                        "vector": blob_to_vector(blob_val),
                        "payload": {
                            "chunk_id": chunk_id,
                            "text": overlap_val or raw_text,
                            "metadata": meta_val if isinstance(meta_val, dict) else {},
                            "token_count": token_val,
                            "provider": provider,
                            "model_name": model_name,
                        },
                    })
                return results

    def count_embeddings(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> int:
        """Đếm tổng số bản ghi vector đã nhúng."""
        with self._lock:
            with self._session_factory() as session:
                query = session.query(ChunkEmbedding)
                if provider:
                    query = query.filter(ChunkEmbedding.provider == provider)
                if model_name:
                    query = query.filter(ChunkEmbedding.model_name == model_name)
                if dimension:
                    query = query.filter(ChunkEmbedding.dimension == dimension)
                return query.count()


# ==============================================================================
# ADAPTER KẾT HỢP TRIỂN KHAI SQL DATABASE PORT
# ==============================================================================

class SqlDatabaseAdapter(SqlDatabasePort):
    """
    Adapter kết nối tới SQL Database (SQLite/SQLAlchemy) hiện thực hóa SqlDatabasePort.
    Cung cấp giao diện thống nhất cho cả ChunkRepository và ChunkEmbeddingRepository.
    """

    def __init__(self, database_url: str, echo: bool = False) -> None:
        self._database_url = database_url
        connect_args = {"timeout": 60} if database_url.startswith("sqlite") else {}
        self._engine = create_engine(
            database_url,
            echo=echo,
            connect_args=connect_args,
            json_serializer=lambda obj: json.dumps(obj, ensure_ascii=False),
        )
        self._session_factory = sessionmaker(
            autocommit=False,
            autoflush=False,
            bind=self._engine,
        )
        self.chunk_repo = ChunkRepository(session_factory=self._session_factory)
        self.embedding_repo = ChunkEmbeddingRepository(session_factory=self._session_factory)

    # 1. KHỞI TẠO & KIỂM TRA TRẠNG THÁI
    def init_db(self) -> None:
        """Khởi tạo các bảng (chunk, chunk_embedding) nếu chưa tồn tại."""
        Base.metadata.create_all(bind=self._engine)

    def health_check(self) -> bool:
        """Kiểm tra kết nối tới cơ sở dữ liệu SQL có đang sẵn sàng không."""
        try:
            with self._session_factory() as session:
                session.execute(text("SELECT 1"))
            return True
        except Exception:
            return False

    # 2. QUẢN LÝ CHUNK
    def fetch_pending(self, limit: Optional[int] = None) -> List[Tuple[str, str]]:
        return self.chunk_repo.fetch_pending(limit=limit)

    def claim_next_pending(self) -> Optional[Tuple[str, str]]:
        return self.chunk_repo.claim_next_pending()

    def reset_processing_to_pending(self) -> int:
        return self.chunk_repo.reset_processing_to_pending()

    def update_provider(self, chunk_id: str, provider: str) -> None:
        self.chunk_repo.update_provider(chunk_id=chunk_id, provider=provider)

    def update_model(self, chunk_id: str, model: str) -> None:
        self.chunk_repo.update_model(chunk_id=chunk_id, model=model)

    def mark_completed(
        self,
        chunk_id: str,
        result: Dict[str, Any],
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        self.chunk_repo.mark_completed(
            chunk_id=chunk_id,
            result=result,
            provider=provider,
            model=model,
        )

    def mark_failed(self, chunk_id: str, max_retries: int) -> str:
        return self.chunk_repo.mark_failed(chunk_id=chunk_id, max_retries=max_retries)

    # 3. QUẢN LÝ VECTOR EMBEDDING
    def fetch_unembedded_chunks(
        self,
        provider: str,
        model_name: str,
        dimension: int,
        limit: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        return self.embedding_repo.fetch_unembedded_chunks(
            provider=provider,
            model_name=model_name,
            dimension=dimension,
            limit=limit,
        )

    def save_embedding(
        self,
        chunk_id: str,
        provider: str,
        model_name: str,
        dimension: int,
        vector: List[float],
        token_count: Optional[int] = None,
    ) -> None:
        self.embedding_repo.save_embedding(
            chunk_id=chunk_id,
            provider=provider,
            model_name=model_name,
            dimension=dimension,
            vector=vector,
            token_count=token_count,
        )

    def save_embeddings_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> int:
        return self.embedding_repo.save_embeddings_batch(records=records)

    def get_embedding(
        self,
        chunk_id: str,
        provider: str,
        model_name: str,
        dimension: int,
    ) -> Optional[Dict[str, Any]]:
        return self.embedding_repo.get_embedding(
            chunk_id=chunk_id,
            provider=provider,
            model_name=model_name,
            dimension=dimension,
        )

    def export_for_qdrant(
        self,
        provider: str,
        model_name: str,
        dimension: int,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        return self.embedding_repo.export_for_qdrant(
            provider=provider,
            model_name=model_name,
            dimension=dimension,
            limit=limit,
            offset=offset,
        )

    def count_embeddings(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> int:
        return self.embedding_repo.count_embeddings(
            provider=provider,
            model_name=model_name,
            dimension=dimension,
        )

