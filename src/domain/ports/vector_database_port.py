from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from src.domain.models import DistanceMetric, VectorRecord, SearchResult


class VectorDatabasePort(ABC):
    """Port trừu tượng cho mọi loại Vector Database."""

    # ==========================================
    # 1. QUẢN LÝ COLLECTION / INDEX (LIFECYCLE)
    # ==========================================

    @abstractmethod
    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE
    ) -> None:
        """Tạo collection/table mới với số chiều và khoảng cách xác định."""
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """Kiểm tra collection có tồn tại hay không."""
        pass

    def get_collection_dimension(self, collection_name: str) -> Optional[int]:
        """Số chiều vector của collection; None nếu adapter không hỗ trợ hoặc không xác định được."""
        return None

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """Xóa hoàn toàn một collection và toàn bộ vector trong đó."""
        pass

    # ==========================================
    # 2. GHI & CẬP NHẬT DỮ LIỆU (WRITE PIPELINE)
    # ==========================================

    @abstractmethod
    def upsert(self, collection_name: str, records: List[VectorRecord]) -> None:
        """Nạp hoặc cập nhật danh sách vector theo lô (batch)."""
        pass

    @abstractmethod
    def delete_by_ids(self, collection_name: str, ids: List[Union[str, int]]) -> None:
        """Xóa các bản ghi theo danh sách ID."""
        pass

    # ==========================================
    # 3. TRUY VẤN DỮ LIỆU (RETRIEVAL)
    # ==========================================

    @abstractmethod
    def similarity_search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None
    ) -> List[SearchResult]:
        """
        Tìm kiếm ngữ nghĩa (Dense Search).
        filters: dictionary chuẩn hóa, ví dụ: {"author": "Minh", "year": 2026}
        """
        pass

    @abstractmethod
    def get_by_id(
        self,
        collection_name: str,
        record_id: Union[str, int]
    ) -> Optional[SearchResult]:
        """Lấy một bản ghi cụ thể theo ID."""
        pass

    # ==========================================
    # 4. TIỆN ÍCH & TRẠNG THÁI (UTILITIES)
    # ==========================================

    @abstractmethod
    def count(self, collection_name: str, filters: Optional[Dict[str, Any]] = None) -> int:
        """Đếm số lượng bản ghi trong collection (hỗ trợ đếm theo filter)."""
        pass

    @abstractmethod
    def health_check(self) -> bool:
        """Kiểm tra kết nối tới dịch vụ cơ sở dữ liệu có đang sẵn sàng không."""
        pass
