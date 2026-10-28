from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional, Dict, Any, Union


class DistanceMetric(str, Enum):
    """Độ đo khoảng cách dùng cho vector search."""
    COSINE = "Cosine"
    EUCLIDEAN = "Euclid"
    DOT = "Dot"
    MANHATTAN = "Manhattan"


@dataclass(frozen=True)
class EmbeddingVector:
    """Value object đại diện cho vector nhúng."""
    values: List[float]
    model_name: str
    dimension: int = field(init=False)
    token_count: Optional[int] = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "dimension", len(self.values))

    def preview(self, head: int = 3, tail: int = 3) -> str:
        """Hiển thị tóm tắt một vài phần tử đầu và cuối của vector."""
        if self.dimension <= head + tail:
            return f"[{', '.join(f'{v:.4f}' for v in self.values)}]"
        head_vals = ", ".join(f"{v:.4f}" for v in self.values[:head])
        tail_vals = ", ".join(f"{v:.4f}" for v in self.values[-tail:])
        return f"[{head_vals}, ... {tail_vals}] (dim: {self.dimension})"


@dataclass
class TextDocument:
    """Thực thể đại diện cho một văn bản cần xử lý embedding."""
    content: str
    id: Optional[str] = None
    embedding: Optional[EmbeddingVector] = None

    def attach_embedding(self, embedding: EmbeddingVector) -> None:
        self.embedding = embedding


@dataclass
class VectorRecord:
    """Bản ghi vector để lưu trữ vào Vector Database."""
    id: Union[str, int]
    vector: List[float]
    payload: Dict[str, Any] = field(default_factory=dict)


# Alias tương thích ngược
VectorPoint = VectorRecord


@dataclass
class SearchResult:
    """Kết quả tìm kiếm tương đồng từ Vector Database."""
    id: Union[str, int]
    score: float
    payload: Dict[str, Any] = field(default_factory=dict)
    vector: Optional[List[float]] = None


