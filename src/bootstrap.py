from dataclasses import dataclass, replace
from pathlib import Path
from typing import Optional

from src.infrastructure.config import AppConfig, get_config
from src.domain.ports.embedding_port import EmbeddingPort
from src.domain.ports.vector_database_port import VectorDatabasePort
from src.domain.ports.sql_database_port import SqlDatabasePort
from src.infrastructure.embedding_adapters.google_ai_studio import GoogleAIStudioEmbeddingAdapter
from src.infrastructure.embedding_adapters.ollama import OllamaEmbeddingAdapter
from src.infrastructure.embedding_adapters.huggingface import HuggingFaceEmbeddingAdapter
from src.infrastructure.vector_database_adapters.qdrant import QdrantVectorDatabaseAdapter
from src.infrastructure.sql_database_adapters.repositories import SqlDatabaseAdapter


@dataclass
class ApplicationContainer:
    """
    Container chứa các đối tượng đã được khởi tạo và tiêm phụ thuộc (Dependency Injection).
    """
    config: AppConfig
    embedding_adapter: EmbeddingPort
    vector_db_adapter: Optional[VectorDatabasePort] = None
    sql_db_adapter: Optional[SqlDatabasePort] = None


def create_vector_database_adapter(config: AppConfig) -> VectorDatabasePort:
    """Factory khởi tạo Vector Database Adapter (Qdrant server)."""
    if not config.qdrant_url:
        raise ValueError("Thiếu qdrant_url: truyền qua entrypoint hoặc đặt QDRANT_URL trong .env.")
    return QdrantVectorDatabaseAdapter(
        url=config.qdrant_url,
        api_key=config.qdrant_api_key,
    )


def create_sql_database_adapter(config: AppConfig) -> SqlDatabasePort:
    """Factory khởi tạo SQL Database Adapter (SQLite qua SQLAlchemy) trên DB đã tồn tại."""
    if not config.sql_db_path:
        raise ValueError("Thiếu sql_db_path: truyền qua entrypoint hoặc đặt SQL_DB_PATH trong .env.")
    if not Path(config.sql_db_path).is_file():
        raise FileNotFoundError(f"Không tìm thấy file SQLite: {config.sql_db_path}")
    adapter = SqlDatabaseAdapter(database_url=f"sqlite:///{config.sql_db_path}")
    adapter.init_db()
    return adapter

def create_embedding_adapter(
        provider: str,
        config: AppConfig,
        model_name: str,
        use_local: bool = False,
) -> EmbeddingPort:
    """Factory khởi tạo adapter theo provider; mọi tham số do entrypoint truyền vào."""
    p = provider.lower().strip()

    if p == "google_ai_studio":
        return GoogleAIStudioEmbeddingAdapter(
            api_key=config.gemini_api_key,
            model_name=model_name,
        )
    elif p == "ollama":
        return OllamaEmbeddingAdapter(
            base_url=config.ollama_base_url,
            model_name=model_name,
            api_key=config.ollama_api_key,
        )
    elif p in ("huggingface", "hf"):
        return HuggingFaceEmbeddingAdapter(
            token=config.hf_token,
            model_name=model_name,
            use_local=use_local,
        )
    raise ValueError(
        f"Không hỗ trợ provider '{provider}'. "
        "Các provider hợp lệ: 'google_ai_studio', 'ollama', 'huggingface'."
    )

def bootstrap(
        provider: str,
        model_name: str,
        sql_db_path: Optional[str] = None,
        qdrant_url: Optional[str] = None,
        use_local: bool = False,
        use_vector_db: bool = True,
        config: Optional[AppConfig] = None,
) -> ApplicationContainer:
    """
    Composition Root: key API lấy từ .env, còn provider/model/đường dẫn DB/URL Qdrant
    do entrypoint truyền vào (được ưu tiên hơn giá trị trong .env).
    Qdrant chỉ được kết nối khi use_vector_db=True.
    """
    app_config = config or get_config()
    overrides = {}
    if sql_db_path:
        overrides["sql_db_path"] = sql_db_path
    if qdrant_url:
        overrides["qdrant_url"] = qdrant_url
    if overrides:
        app_config = replace(app_config, **overrides)

    embedding_adapter = create_embedding_adapter(
        provider=provider,
        config=app_config,
        model_name=model_name,
        use_local=use_local,
    )
    sql_db = create_sql_database_adapter(config=app_config)
    vector_db = create_vector_database_adapter(config=app_config) if use_vector_db else None

    return ApplicationContainer(
        config=app_config,
        embedding_adapter=embedding_adapter,
        vector_db_adapter=vector_db,
        sql_db_adapter=sql_db,
    )