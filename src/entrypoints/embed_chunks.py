"""
Entrypoint nhúng các chunk chưa có vector.
Chạy trực tiếp trong IDE (PyCharm...) sau khi chỉnh các tham số bên dưới.
.env chỉ chứa API key (và có thể SQL_DB_PATH / QDRANT_URL); giá trị khai báo ở đây được ưu tiên.

Mỗi batch được ghi song song vào SQLite (chunk_embedding.vector_data) và Qdrant server.
"""

# ==== Tham số chạy (chỉnh trong IDE) =================================
# Embedding
PROVIDER: str = "huggingface"       # "huggingface" | "ollama" | "google_ai_studio"
MODEL_NAME: str = "bkai-foundation-models/vietnamese-bi-encoder"
DIMENSION: int | None = 768         # None -> tự đo từ model
USE_LOCAL: bool = True              # Chỉ cho huggingface: True = chạy local, False = Inference API

# Cơ sở dữ liệu
SOURCE_DB_PATH: str = r"D:\projects\text2triple\data\historical_knowledge\staging_chunks.db"
QDRANT_URL: str = "http://localhost:6333"
QDRANT_COLLECTION: str = "vnhis"
PUSH_QDRANT: bool = True            # True: ghi song song vào Qdrant

# Chạy
BATCH_SIZE: int = 16
LIMIT: int | None = None               # Giới hạn số chunk (để thử); None = nhúng toàn bộ
MAX_RETRIES: int = 3
# ======================================================================

from src.bootstrap import bootstrap
from src.application.use_cases.embed_chunks import EmbedChunksUseCase, EmbedStats


def run() -> EmbedStats:
    container = bootstrap(
        provider=PROVIDER,
        model_name=MODEL_NAME,
        sql_db_path=SOURCE_DB_PATH,
        qdrant_url=QDRANT_URL,
        use_local=USE_LOCAL,
        use_vector_db=PUSH_QDRANT,
    )

    # Tiêm các port vào UseCase theo Clean Architecture
    use_case = EmbedChunksUseCase(
        sql_db=container.sql_db_adapter,
        embedding_service=container.embedding_adapter,
        vector_db=container.vector_db_adapter,
    )

    stats: EmbedStats = use_case.execute(
        provider=PROVIDER,
        model_name=MODEL_NAME,
        batch_size=BATCH_SIZE,
        limit=LIMIT,
        dimension=DIMENSION,
        max_retries=MAX_RETRIES,
        push_qdrant=PUSH_QDRANT,
        qdrant_collection=QDRANT_COLLECTION,
    )
    print(
        f"Hoan thanh. Da nhung {stats.successful}/{stats.total} chunk, "
        f"that bai {stats.failed}."
    )
    if stats.failed_ids:
        print("Chunk that bai:", stats.failed_ids)
    if stats.qdrant_failed_ids:
        print("Chunk da luu SQLite nhung loi Qdrant:", stats.qdrant_failed_ids)
    return stats


if __name__ == "__main__":
    run()
