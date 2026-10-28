import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


def _load_dotenv() -> None:
    """Đọc file .env nếu tồn tại mà không bắt buộc phải cài thư viện ngoài."""
    env_path = Path(".env")
    if not env_path.exists():
        return
    try:
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip("'\"")
                if k not in os.environ:
                    os.environ[k] = v
    except Exception:
        pass


_load_dotenv()

# Đường dẫn gốc dự án và thư mục data
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"


@dataclass
class AppConfig:
    """
    Cấu hình hệ thống: CHỈ chứa API key và đường dẫn/URL cơ sở dữ liệu lấy từ env.
    Provider, model, số chiều, chế độ local/online... được truyền từ entrypoint.
    """
    # API keys
    gemini_api_key: Optional[str] = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    hf_token: Optional[str] = os.getenv("HF_TOKEN")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY")
    ollama_api_key: Optional[str] = os.getenv("OLLAMA_API_KEY")

    # Địa chỉ Ollama (nếu không đặt, adapter dùng Ollama online https://ollama.com)
    ollama_base_url: Optional[str] = os.getenv("OLLAMA_BASE_URL") or os.getenv("OLLAMA_HOST")

    # Thư mục dự án / data
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR

    # Cơ sở dữ liệu: tùy chọn đặt trong .env; giá trị truyền từ entrypoint được ưu tiên
    qdrant_url: Optional[str] = os.getenv("QDRANT_URL")
    sql_db_path: Optional[str] = os.getenv("SQL_DB_PATH")


def get_config() -> AppConfig:
    """Trả về cấu hình hiện tại của ứng dụng."""
    return AppConfig()
