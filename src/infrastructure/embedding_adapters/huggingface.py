import json
import urllib.request
import urllib.error
from typing import List, Optional
from src.domain.ports.embedding_port import EmbeddingPort
from src.domain.models import EmbeddingVector
from src.domain.exceptions import (
    EmbeddingServiceError,
    ProviderConnectionError,
    MissingApiKeyError,
)


class HuggingFaceEmbeddingAdapter(EmbeddingPort):
    """
    Adapter cho hạ tầng Hugging Face:
    1. Chế độ Cloud Inference API (mặc định): Gọi trực tiếp Hugging Face Inference API qua HTTP.
       Hỗ trợ bất kỳ model nào như 'sentence-transformers/all-MiniLM-L6-v2', 'BAAI/bge-m3'.
    2. Chế độ Local: Nếu use_local=True, sử dụng thư viện sentence-transformers nếu có cài đặt.
    """

    DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
    API_URL_TEMPLATE = "https://api-inference.huggingface.co/pipeline/feature-extraction/{model_name}"

    def __init__(
        self,
        token: Optional[str] = None,
        model_name: str = DEFAULT_MODEL,
        use_local: bool = False,
    ) -> None:
        self._token = token
        self._model_name = model_name
        self._use_local = use_local
        self._local_model = None

        if self._use_local:
            self._init_local_model()

    def count_tokens(self, text: str) -> int:
        """Đếm số token bằng tokenizer cục bộ (nếu có) hoặc ước tính chuẩn."""
        clean = text.strip()
        if not clean:
            return 0
        if self._local_model and hasattr(self._local_model, "tokenizer"):
            try:
                return len(self._local_model.tokenizer.encode(clean))
            except Exception:
                pass
        return max(1, int(len(clean.split()) * 1.3))

    def _init_local_model(self) -> None:
        try:
            from sentence_transformers import SentenceTransformer
            self._local_model = SentenceTransformer(self._model_name)
        except ImportError:
            raise EmbeddingServiceError(
                "Chế độ HF Local yêu cầu cài đặt gói 'sentence-transformers'. "
                "Vui lòng chạy: pip install sentence-transformers"
            )
        except Exception as e:
            raise EmbeddingServiceError(f"Không thể tải model Hugging Face cục bộ '{self._model_name}': {e}") from e

    def embed_text(self, text: str) -> EmbeddingVector:
        vectors = self.embed_batch([text])
        if not vectors:
            raise EmbeddingServiceError("Hugging Face không trả về vector cho văn bản.")
        return vectors[0]

    def embed_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        if self._use_local and self._local_model:
            return self._embed_batch_local(texts)
        return self._embed_batch_api(texts)

    def _embed_batch_local(self, texts: List[str]) -> List[EmbeddingVector]:
        try:
            embeddings = self._local_model.encode(texts, convert_to_numpy=True)
            return [
                EmbeddingVector(
                    values=emb.tolist(),
                    model_name=self._model_name,
                    token_count=self.count_tokens(texts[idx]),
                )
                for idx, emb in enumerate(embeddings)
            ]
        except Exception as e:
            raise EmbeddingServiceError(f"Lỗi khi thực thi local model Hugging Face: {e}") from e

    def _embed_batch_api(self, texts: List[str]) -> List[EmbeddingVector]:
        url = self.API_URL_TEMPLATE.format(model_name=self._model_name)
        payload = {
            "inputs": texts,
            "options": {"wait_for_model": True}
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"

        req = urllib.request.Request(
            url,
            data=data,
            headers=headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                
                # API HF trả về:
                # - list[float] nếu single input
                # - list[list[float]] nếu batch input
                if isinstance(resp_data, list) and resp_data and isinstance(resp_data[0], list):
                    return [
                        EmbeddingVector(
                            values=vec,
                            model_name=self._model_name,
                            token_count=self.count_tokens(texts[idx]),
                        )
                        for idx, vec in enumerate(resp_data)
                    ]
                elif isinstance(resp_data, list) and resp_data and isinstance(resp_data[0], (int, float)):
                    return [
                        EmbeddingVector(
                            values=resp_data,
                            model_name=self._model_name,
                            token_count=self.count_tokens(texts[0]),
                        )
                    ]
                else:
                    raise EmbeddingServiceError(f"Định dạng phản hồi từ Hugging Face không hợp lệ: {resp_data}")

        except urllib.error.HTTPError as e:
            err_body = e.read().decode("utf-8", errors="replace")
            if e.code == 401:
                raise MissingApiKeyError(
                    "Token Hugging Face không hợp lệ hoặc bị thiếu. "
                    "Vui lòng cấu hình HF_TOKEN trong môi trường."
                ) from e
            if e.code == 503:
                raise EmbeddingServiceError("Model Hugging Face đang tải (503). Hãy thử lại sau giây lát.") from e
            raise EmbeddingServiceError(f"Lỗi Hugging Face API [HTTP {e.code}]: {err_body}") from e
        except urllib.error.URLError as e:
            raise ProviderConnectionError(f"Không thể kết nối đến Hugging Face API: {e.reason}") from e
        except Exception as e:
            if isinstance(e, EmbeddingServiceError):
                raise
            raise EmbeddingServiceError(f"Lỗi khi gọi Hugging Face: {str(e)}") from e
