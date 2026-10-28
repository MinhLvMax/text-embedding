from abc import ABC, abstractmethod
from typing import List
from src.domain.models import EmbeddingVector


class EmbeddingPort(ABC):
    """
    Port giao tiếp chuẩn cho các dịch vụ nhúng văn bản (Embedding).
    Tất cả các adapter (Google AI Studio, Ollama, Hugging Face, ...)
    đều phải hiện thực hóa interface này.
    """

    @abstractmethod
    def count_tokens(self, text: str) -> int:
        """
        Tính số lượng token của một đoạn văn bản theo tokenizer của model.

        :param text: Chuỗi văn bản cần đếm token.
        :return: Số lượng token (int).
        """
        pass

    @abstractmethod
    def embed_text(self, text: str) -> EmbeddingVector:
        """
        Nhúng một đoạn văn bản thành vector embedding.

        :param text: Chuỗi văn bản cần nhúng.
        :return: EmbeddingVector chứa danh sách số thực và metadata.
        """
        pass

    @abstractmethod
    def embed_batch(self, texts: List[str]) -> List[EmbeddingVector]:
        """
        Nhúng một danh sách các văn bản thành danh sách vector embedding.

        :param texts: Danh sách các chuỗi văn bản.
        :return: Danh sách các EmbeddingVector tương ứng.
        """
        pass
