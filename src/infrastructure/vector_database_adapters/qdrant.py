import uuid
from typing import List, Optional, Dict, Any, Union

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    PointIdsList,
)
from qdrant_client.http.exceptions import UnexpectedResponse

from src.domain.ports.vector_database_port import VectorDatabasePort
from src.domain.models import DistanceMetric, VectorRecord, SearchResult
from src.domain.exceptions import ProviderConnectionError, EmbeddingServiceError


class QdrantVectorDatabaseAdapter(VectorDatabasePort):
    """
    Adapter kết nối đến Qdrant Vector Database sử dụng thư viện chính thức 'qdrant-client'.
    Hiện thực hóa VectorDatabasePort theo chuẩn Clean Architecture.
    """

    def __init__(self, url: str, api_key: Optional[str] = None) -> None:
        """Kết nối tới Qdrant server qua `url` (kèm `api_key` nếu có)."""
        if not url:
            raise ProviderConnectionError("Cần cung cấp url của Qdrant server.")
        self._url = url.rstrip("/")
        self._api_key = api_key
        try:
            self._client = QdrantClient(url=self._url, api_key=self._api_key, timeout=30)
        except Exception as e:
            raise ProviderConnectionError(f"Không thể khởi tạo QdrantClient tại '{self._url}': {e}") from e

    def _to_valid_qdrant_id(self, point_id: Union[str, int]) -> Union[str, int]:
        """
        Qdrant yêu cầu ID là số nguyên dương hoặc chuỗi UUID hợp lệ.
        Nếu ID là chuỗi không phải UUID (ví dụ 'doc_001'), tự sinh UUID xác định (uuid5).
        """
        if isinstance(point_id, int):
            return point_id
        try:
            uuid.UUID(str(point_id))
            return str(point_id)
        except ValueError:
            return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(point_id)))

    def _build_filter(self, filters: Optional[Dict[str, Any]]) -> Optional[Filter]:
        """Chuyển đổi dictionary filter thành đối tượng Filter của qdrant-client."""
        if not filters:
            return None
        conditions = [
            FieldCondition(key=k, match=MatchValue(value=v))
            for k, v in filters.items()
        ]
        return Filter(must=conditions)

    # ==========================================
    # 1. QUẢN LÝ COLLECTION / INDEX (LIFECYCLE)
    # ==========================================

    def create_collection(
        self,
        collection_name: str,
        dimension: int,
        metric: DistanceMetric = DistanceMetric.COSINE,
    ) -> None:
        """Tạo collection/table mới với số chiều và khoảng cách xác định."""
        try:
            if self.collection_exists(collection_name):
                return

            metric_map = {
                DistanceMetric.COSINE: Distance.COSINE,
                DistanceMetric.EUCLIDEAN: Distance.EUCLID,
                DistanceMetric.DOT: Distance.DOT,
                DistanceMetric.MANHATTAN: Distance.MANHATTAN,
            }
            dist = metric_map.get(metric, Distance.COSINE)

            self._client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=dimension, distance=dist),
            )
        except Exception as e:
            raise EmbeddingServiceError(f"Lỗi tạo collection '{collection_name}': {e}") from e

    def collection_exists(self, collection_name: str) -> bool:
        """Kiểm tra collection có tồn tại hay không."""
        try:
            return self._client.collection_exists(collection_name=collection_name)
        except Exception:
            return False

    def get_collection_dimension(self, collection_name: str) -> Optional[int]:
        """Trả về số chiều vector của collection (None nếu không xác định được)."""
        try:
            info = self._client.get_collection(collection_name=collection_name)
            vectors = info.config.params.vectors
            if isinstance(vectors, dict):
                vectors = next(iter(vectors.values()), None)
            return int(vectors.size) if vectors is not None else None
        except Exception:
            return None

    def delete_collection(self, collection_name: str) -> None:
        """Xóa hoàn toàn một collection và toàn bộ vector trong đó."""
        try:
            if self.collection_exists(collection_name):
                self._client.delete_collection(collection_name=collection_name)
        except Exception as e:
            raise EmbeddingServiceError(f"Lỗi xóa collection '{collection_name}': {e}") from e

    # ==========================================
    # 2. GHI & CẬP NHẬT DỮ LIỆU (WRITE PIPELINE)
    # ==========================================

    def upsert(self, collection_name: str, records: List[VectorRecord]) -> None:
        """Nạp hoặc cập nhật danh sách vector theo lô (batch)."""
        if not records:
            return

        try:
            points: List[PointStruct] = []
            for r in records:
                q_id = self._to_valid_qdrant_id(r.id)
                payload = dict(r.payload)
                if str(q_id) != str(r.id):
                    payload["_original_id"] = str(r.id)

                points.append(
                    PointStruct(
                        id=q_id,
                        vector=r.vector,
                        payload=payload,
                    )
                )

            self._client.upsert(
                collection_name=collection_name,
                points=points,
                wait=True,
            )
        except Exception as e:
            raise EmbeddingServiceError(f"Lỗi upsert vào collection '{collection_name}': {e}") from e

    def delete_by_ids(self, collection_name: str, ids: List[Union[str, int]]) -> None:
        """Xóa các bản ghi theo danh sách ID."""
        if not ids:
            return

        try:
            valid_ids = [self._to_valid_qdrant_id(i) for i in ids]
            self._client.delete(
                collection_name=collection_name,
                points_selector=PointIdsList(points=valid_ids),
                wait=True,
            )
        except Exception as e:
            raise EmbeddingServiceError(f"Lỗi xóa bản ghi trong '{collection_name}': {e}") from e

    # ==========================================
    # 3. TRUY VẤN DỮ LIỆU (RETRIEVAL)
    # ==========================================

    def similarity_search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        """Tìm kiếm ngữ nghĩa (Dense Search) kèm bộ lọc filters."""
        try:
            q_filter = self._build_filter(filters)
            scored_points = self._client.query_points(
                collection_name=collection_name,
                query=query_vector,
                limit=top_k,
                query_filter=q_filter,
                score_threshold=score_threshold,
                with_payload=True,
                with_vectors=True,
            ).points

            results: List[SearchResult] = []
            for sp in scored_points:
                p_load = dict(sp.payload) if sp.payload else {}
                orig_id = p_load.get("_original_id", sp.id)
                results.append(
                    SearchResult(
                        id=orig_id,
                        score=float(sp.score),
                        payload=p_load,
                        vector=sp.vector if isinstance(sp.vector, list) else None,
                    )
                )
            return results
        except Exception as e:
            raise EmbeddingServiceError(f"Lỗi tìm kiếm tương đồng trên '{collection_name}': {e}") from e

    def get_by_id(
        self,
        collection_name: str,
        record_id: Union[str, int],
    ) -> Optional[SearchResult]:
        """Lấy một bản ghi cụ thể theo ID."""
        try:
            q_id = self._to_valid_qdrant_id(record_id)
            retrieved = self._client.retrieve(
                collection_name=collection_name,
                ids=[q_id],
                with_payload=True,
                with_vectors=True,
            )
            if not retrieved:
                return None

            point = retrieved[0]
            p_load = dict(point.payload) if point.payload else {}
            orig_id = p_load.get("_original_id", point.id)

            return SearchResult(
                id=orig_id,
                score=1.0,
                payload=p_load,
                vector=point.vector if isinstance(point.vector, list) else None,
            )
        except Exception as e:
            raise EmbeddingServiceError(f"Lỗi truy vấn ID '{record_id}' từ '{collection_name}': {e}") from e

    # ==========================================
    # 4. TIỆN ÍCH & TRẠNG THÁI (UTILITIES)
    # ==========================================

    def count(self, collection_name: str, filters: Optional[Dict[str, Any]] = None) -> int:
        """Đếm số lượng bản ghi trong collection (hỗ trợ đếm theo filter)."""
        try:
            q_filter = self._build_filter(filters)
            res = self._client.count(
                collection_name=collection_name,
                count_filter=q_filter,
                exact=True,
            )
            return res.count
        except Exception as e:
            raise EmbeddingServiceError(f"Lỗi đếm số lượng bản ghi trong '{collection_name}': {e}") from e

    def health_check(self) -> bool:
        """Kiểm tra kết nối tới dịch vụ cơ sở dữ liệu có đang sẵn sàng không."""
        try:
            # Gọi API lấy danh sách collections để xác thực kết nối
            self._client.get_collections()
            return True
        except Exception:
            return False
