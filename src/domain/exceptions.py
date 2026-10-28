class DomainException(Exception):
    """Ngoại lệ cơ sở cho tầng Domain."""
    pass


class EmptyContentError(DomainException):
    """Ngoại lệ khi văn bản đầu vào trống."""
    def __init__(self, message: str = "Nội dung văn bản không được để trống."):
        super().__init__(message)


class EmbeddingServiceError(DomainException):
    """Ngoại lệ khi có lỗi xảy ra trong quá trình sinh embedding từ provider."""
    pass


class MissingApiKeyError(EmbeddingServiceError):
    """Ngoại lệ khi thiếu API key cấu hình cho provider."""
    pass


class ProviderConnectionError(EmbeddingServiceError):
    """Ngoại lệ khi không thể kết nối tới server embedding provider (ví dụ Ollama)."""
    pass
