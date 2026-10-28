import json
import urllib.request
import urllib.error
from typing import List, Optional
from src.domain.ports.embedding_port import EmbeddingPort
from src.domain.models import EmbeddingVector
from src.domain.exceptions import (
    EmbeddingServiceError,
    MissingApiKeyError,
    ProviderConnectionError,
)


class GoogleAIStudioEmbeddingAdapter(EmbeddingPort):
    """
    Adapter kết nối đến Google AI Studio (Gemini Embedding API).
    Mặc định sử dụng model 'text-embedding-004'.
    Sử dụng urllib.request để hoạt động ngay mà không phụ thuộc gói bên ngoài.
    """

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: Optional[str] = None, model_name: str = "text-embedding-004") -> None:
        self._api_key = api_key
        # Đảm bảo tên model chuẩn dạng 'models/<name>' hoặc giữ nguyên nếu đã có prefix
        self._model_name = model_name
        self._endpoint_model = model_name if model_name.startswith("models/") else f"models/{model_name}"

    def _ensure_api_key(self) -> str:
        if not self._api_key:
            raise MissingApiKeyError(
                "Chưa cấu hình API Key cho Google AI Studio. "
                "Vui lòng thiết lập biến môi trường GEMINI_API_KEY hoặc GOOGLE_API_KEY."
            )
        return self._api_key

    def count_tokens(self, text: str) -> int:
        """Tính số lượng token chính xác từ Google AI Studio (:countTokens)."""
        api_key = self._ensure_api_key()
        url = f"{self.BASE_URL}/{self._endpoint_model}:countTokens?key={api_key}"
        payload = {
            "contents": [{
                "parts": [{"text": text}]
            }]
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=15) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                return resp_data.get("totalTokens", max(1, int(len(text.split()) * 1.3)))
        except Exception:
            # Fallback nếu mạng chậm hoặc lỗi API countTokens
            return max(1, int(len(text.split()) * 1.3))

    def embed_text(self, text: str) -> EmbeddingVector:
        api_key = self._ensure_api_key()
        url = f"{self.BASE_URL}/{self._endpoint_model}:embedContent?key={api_key}"

        payload = {
            "model": self._endpoint_model,
            "content": {
                "parts": [{"text": text}]
            }
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                values = resp_data.get("embedding", {}).get("values", [])
                if not values:
                    raise EmbeddingServiceError("Không nhận được vector từ Google AI Studio.")
                tokens = self.count_tokens(text)
                return EmbeddingVector(values=values, model_name=self._model_name, token_count=tokens)
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise EmbeddingServiceError(f"Lỗi Google AI Studio API [HTTP {e.code}]: {err_body}") from e
        except urllib.error.URLError as e:
            raise ProviderConnectionError(f"Không thể kết nối đến Google AI Studio: {e.reason}") from e
        except Exception as e:
            if isinstance(e, EmbeddingServiceError):
                raise
            raise EmbeddingServiceError(f"Lỗi không xác định khi gọi Google AI Studio: {str(e)}") from e

    def embed_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        api_key = self._ensure_api_key()
        url = f"{self.BASE_URL}/{self._endpoint_model}:batchEmbedContents?key={api_key}"

        requests_payload = [
            {
                "model": self._endpoint_model,
                "content": {"parts": [{"text": t}]}
            }
            for t in texts
        ]
        payload = {"requests": requests_payload}
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                raw_embeddings = resp_data.get("embeddings", [])
                if len(raw_embeddings) != len(texts):
                    raise EmbeddingServiceError(
                        f"Số lượng vector trả về ({len(raw_embeddings)}) không khớp với số văn bản ({len(texts)})."
                    )
                return [
                    EmbeddingVector(
                        values=item.get("values", []),
                        model_name=self._model_name,
                        token_count=max(1, int(len(texts[idx].split()) * 1.3)),
                    )
                    for idx, item in enumerate(raw_embeddings)
                ]
        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            raise EmbeddingServiceError(f"Lỗi Google AI Studio API [HTTP {e.code}]: {err_body}") from e
        except urllib.error.URLError as e:
            raise ProviderConnectionError(f"Không thể kết nối đến Google AI Studio: {e.reason}") from e
        except Exception as e:
            if isinstance(e, EmbeddingServiceError):
                raise
            raise EmbeddingServiceError(f"Lỗi không xác định khi gọi batch Google AI Studio: {str(e)}") from e
