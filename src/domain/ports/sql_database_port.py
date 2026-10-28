from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class SqlDatabasePort(ABC):
    """Port trừu tượng cho cơ sở dữ liệu quan hệ (SQL Database)."""

    # ==========================================
    # 1. KHỞI TẠO & KIỂM TRA TRẠNG THÁI (LIFECYCLE & UTILITIES)
    # ==========================================

    @abstractmethod
    def init_db(self) -> None:
        """Khởi tạo các bảng trong cơ sở dữ liệu nếu chưa tồn tại."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Kiểm tra kết nối tới cơ sở dữ liệu SQL có đang sẵn sàng không."""
        pass

    # ==========================================
    # 2. QUẢN LÝ CHUNK (CHUNK REPOSITORY OPERATIONS)
    # ==========================================

    @abstractmethod
    def fetch_pending(self, limit: Optional[int] = None) -> List[Tuple[str, str]]:
        """Trả về [(chunk_id, text)] của các chunk đang ở trạng thái PENDING."""
        pass

    @abstractmethod
    def claim_next_pending(self) -> Optional[Tuple[str, str]]:
        """Lấy nguyên tử (atomic) 1 chunk PENDING và chuyển sang PROCESSING."""
        pass

    @abstractmethod
    def reset_processing_to_pending(self) -> int:
        """Khôi phục các chunk đang bị kẹt ở trạng thái PROCESSING về lại PENDING."""
        pass

    @abstractmethod
    def update_provider(self, chunk_id: str, provider: str) -> None:
        """Cập nhật tên nhà cung cấp (provider) thực hiện xử lý cho chunk."""
        pass

    @abstractmethod
    def update_model(self, chunk_id: str, model: str) -> None:
        """Cập nhật tên model thực hiện xử lý cho chunk."""
        pass

    @abstractmethod
    def mark_completed(
        self,
        chunk_id: str,
        result: Dict[str, Any],
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        """Đánh dấu chunk đã xử lý thành công và lưu kết quả."""
        pass

    @abstractmethod
    def mark_failed(self, chunk_id: str, max_retries: int) -> str:
        """Tăng retry_count; chuyển sang FAILED nếu vượt ngưỡng, ngược lại giữ PENDING."""
        pass

    # ==========================================
    # 3. QUẢN LÝ VECTOR EMBEDDING (CHUNK EMBEDDING OPERATIONS)
    # ==========================================

    @abstractmethod
    def fetch_unembedded_chunks(
        self,
        provider: str,
        model_name: str,
        dimension: int,
        limit: Optional[int] = None,
    ) -> List[Tuple[str, str]]:
        """Lấy danh sách (chunk_id, text) chưa có vector nhúng cho (provider, model_name, dimension)."""
        pass

    @abstractmethod
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
        pass

    @abstractmethod
    def save_embeddings_batch(
        self,
        records: List[Dict[str, Any]],
    ) -> int:
        """Lưu hàng loạt (batch UPSERT) danh sách vector nhúng."""
        pass

    @abstractmethod
    def get_embedding(
        self,
        chunk_id: str,
        provider: str,
        model_name: str,
        dimension: int,
    ) -> Optional[Dict[str, Any]]:
        """Lấy vector nhúng và metadata đã giải mã của một chunk."""
        pass

    @abstractmethod
    def export_for_qdrant(
        self,
        provider: str,
        model_name: str,
        dimension: int,
        limit: Optional[int] = None,
        offset: int = 0,
    ) -> List[Dict[str, Any]]:
        """Xuất danh sách vector kết hợp payload từ bảng chunk để đẩy sang Qdrant."""
        pass

    @abstractmethod
    def count_embeddings(
        self,
        provider: Optional[str] = None,
        model_name: Optional[str] = None,
        dimension: Optional[int] = None,
    ) -> int:
        """Đếm tổng số bản ghi vector đã nhúng."""
        pass
