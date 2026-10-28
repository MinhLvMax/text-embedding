import re
from datetime import datetime
from typing import Any, Dict, Optional
from sqlalchemy import String, Text, JSON, DateTime, Integer, LargeBinary, ForeignKey, UniqueConstraint, Index, func, create_engine
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, declared_attr, sessionmaker, MappedAsDataclass


class Base(MappedAsDataclass, DeclarativeBase):
    @declared_attr.directive
    def __tablename__(cls) -> str:
        # Regex chuyển CamelCase thành snake_case
        name = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()
        return name


class Chunk(Base):
    # 1. Khóa chính định danh chunk
    chunk_id: Mapped[str] = mapped_column(
        String(100),
        primary_key=True,
        comment="Mã định danh duy nhất của chunk"
    )

    # 2. Nội dung văn bản thô
    text: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        comment="Nội dung văn bản để gửi cho LLM hoặc Embedding"
    )

    # 3. Metadata linh hoạt cho mọi loại sách/tài liệu
    chunk_metadata: Mapped[Dict[str, Any]] = mapped_column(
        JSON,
        nullable=True,
        default=dict,
        comment="Chứa toàn bộ thông tin phụ: book_name, page, author, words, sentences..."
    )

    # Các trường quản trị vòng đời (rất hữu ích cho hàng đợi worker)
    status: Mapped[str] = mapped_column(
        String(20),
        default="PENDING",
        index=True,
        comment="PENDING, PROCESSING, COMPLETED, FAILED"
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        init=False
    )

    retry_count: Mapped[int] = mapped_column(default=0)

    graph_extraction: Mapped[Optional[Dict[str, Any]]] = mapped_column(
        JSON,
        nullable=True,
        default=dict,
        comment="Kết quả trích xuất thực thể và quan hệ"
    )

    chunk_overlap_200: Mapped[Optional[str]] = mapped_column(
        Text,
        nullable=True,
        default=None,
        comment="Text của chunk kèm 200 ký tự cuối trang trước và 200 ký tự đầu trang sau"
    )

    provider: Mapped[Optional[str]] = mapped_column(
        String(50),
        nullable=True,
        default=None,
        comment="Nhà cung cấp LLM: openai, groq, ollama, google"
    )

    model: Mapped[Optional[str]] = mapped_column(
        String(100),
        nullable=True,
        default=None,
        comment="Tên model thực hiện trích xuất"
    )


class ChunkEmbedding(Base):
    id: Mapped[int] = mapped_column(
        Integer,
        primary_key=True,
        autoincrement=True,
        init=False
    )

    # 1. Định danh liên kết bảng chunk
    chunk_id: Mapped[str] = mapped_column(
        String(100),
        ForeignKey("chunk.chunk_id", ondelete="CASCADE"),
        index=True,
        comment="Mã định danh liên kết bảng chunk"
    )

    # 2. Thông số nhà cung cấp, model và kích thước vector
    provider: Mapped[str] = mapped_column(
        String(50),
        comment="ollama, huggingface, openai, google_ai_studio..."
    )

    model_name: Mapped[str] = mapped_column(
        String(100),
        comment="bge-m3, all-MiniLM-L6-v2, text-embedding-004..."
    )

    dimension: Mapped[int] = mapped_column(
        Integer,
        comment="384, 768, 1024, 1536..."
    )

    # 3. Dữ liệu vector (BLOB nhị phân float32)
    vector_data: Mapped[bytes] = mapped_column(
        LargeBinary,
        comment="Dữ liệu vector lưu dạng BLOB (float32)"
    )

    # 4. Số token phục vụ thống kê chi phí / độ dài
    token_count: Mapped[Optional[int]] = mapped_column(
        Integer,
        nullable=True,
        default=None,
        comment="Số token phục vụ thống kê chi phí / độ dài"
    )

    # 5. Thời gian ghi nhận
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        init=False
    )

    __table_args__ = (
        UniqueConstraint(
            "chunk_id", "provider", "model_name", "dimension",
            name="uq_chunk_provider_model_dim"
        ),
        Index("idx_chunk_embedding_lookup", "provider", "model_name", "dimension"),
        Index("idx_chunk_embedding_chunk_id", "chunk_id"),
    )
