import json
import urllib.request
import urllib.error
from typing import List, Optional
from src.domain.ports.embedding_port import EmbeddingPort
from src.domain.models import EmbeddingVector
from src.domain.exceptions import (
    EmbeddingServiceError,
    ProviderConnectionError,
)


class OllamaEmbeddingAdapter(EmbeddingPort):
    """
    Adapter kết nối đến local Ollama Server (mặc định http://localhost:11434).
    Hỗ trợ endpoint /api/embed (chuẩn mới nhất) và fallback /api/embeddings.
    """

    DEFAULT_BASE_URL = "https://ollama.com"

    def __init__(self, base_url: Optional[str] = None, model_name: str = "nomic-embed-text",
                 api_key: Optional[str] = None) -> None:
        self._base_url = (base_url or self.DEFAULT_BASE_URL).rstrip("/")
        self._model_name = model_name
        self._headers = {"Content-Type": "application/json"}
        if api_key:
            self._headers["Authorization"] = f"Bearer {api_key}"

    def count_tokens(self, text: str) -> int:
        """Ước tính số token cho model Ollama theo BPE tokenization heuristic (~1.3 token/từ)."""
        clean = text.strip()
        if not clean:
            return 0
        words = clean.split()
        return max(1, int(len(words) * 1.3))

    def embed_text(self, text: str) -> EmbeddingVector:
        vectors = self.embed_batch([text])
        if not vectors:
            raise EmbeddingServiceError("Ollama không trả về vector cho văn bản.")
        return vectors[0]

    def embed_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        # Thử endpoint chuẩn mới /api/embed
        url = f"{self._base_url}/api/embed"
        payload = {
            "model": self._model_name,
            "input": texts if len(texts) > 1 else texts[0]
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers=self._headers,
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                resp_data = json.loads(resp.read().decode("utf-8"))
                embeddings = resp_data.get("embeddings", [])
                if not embeddings and "embedding" in resp_data:
                    embeddings = [resp_data["embedding"]]

                prompt_eval_count = resp_data.get("prompt_eval_count")
                # Nếu chỉ nhúng 1 câu và có prompt_eval_count từ Ollama, ưu tiên dùng chính xác
                total_embs = len(embeddings)
                
                return [
                    EmbeddingVector(
                        values=v,
                        model_name=self._model_name,
                        token_count=(
                            prompt_eval_count if (total_embs == 1 and prompt_eval_count)
                            else self.count_tokens(texts[i])
                        ),
                    )
                    for i, v in enumerate(embeddings)
                ]
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Fallback sang endpoint cũ /api/embeddings từng cái một
                return self._embed_batch_legacy(texts)
            err_body = e.read().decode("utf-8", errors="replace")
            raise EmbeddingServiceError(f"Lỗi Ollama API [HTTP {e.code}]: {err_body}") from e
        except urllib.error.URLError as e:
            raise ProviderConnectionError(
                f"Không thể kết nối đến Ollama tại '{self._base_url}'. "
                "Vui lòng đảm bảo Ollama server đã được khởi động ('ollama serve')."
            ) from e
        except Exception as e:
            if isinstance(e, EmbeddingServiceError):
                raise
            raise EmbeddingServiceError(f"Lỗi khi gọi Ollama: {str(e)}") from e

    def _embed_batch_legacy(self, texts: List[str]) -> List[EmbeddingVector]:
        """Fallback cho các phiên bản Ollama cũ hơn dùng /api/embeddings."""
        url = f"{self._base_url}/api/embeddings"
        results: List[EmbeddingVector] = []
        for text in texts:
            payload = {"model": self._model_name, "prompt": text}
            data = json.dumps(payload).encode("utf-8")
            req = urllib.request.Request(
                url,
                data=data,
                headers=self._headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(req, timeout=30) as resp:
                    resp_data = json.loads(resp.read().decode("utf-8"))
                    val = resp_data.get("embedding", [])
                    results.append(EmbeddingVector(values=val, model_name=self._model_name))
            except Exception as e:
                raise EmbeddingServiceError(f"Lỗi fallback Ollama: {str(e)}") from e
        return results
