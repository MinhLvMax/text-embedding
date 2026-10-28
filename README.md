# Text Embedding - Clean Architecture

Dự án nhúng văn bản (Text Embedding) được thiết kế theo mô hình Clean Architecture / Hexagonal Architecture trong thư mục `src/`.

## 📁 Cấu trúc thư mục

```
src/
├── __init__.py
├── bootstrap.py                 # Composition Root: Ráp nối cấu hình, adapters và use cases
├── domain/                      # Tầng Domain (Nghiệp vụ cốt lõi, độc lập)
│   ├── __init__.py
│   ├── exceptions.py            # Ngoại lệ nghiệp vụ
│   ├── models.py                # EmbeddingVector, TextDocument
│   └── ports/                   # Cổng giao tiếp trừu tượng
│       ├── __init__.py
│       └── embedding_port.py    # EmbeddingPort (embed_text, embed_batch)
├── application/                 # Tầng Application (Use cases & DTOs)
│   ├── __init__.py
│   ├── dtos.py                  # Request & Response DTOs
│   └── use_cases/
│       ├── __init__.py
│       └── embed_text_use_case.py
├── infrastructure/              # Tầng Infrastructure (Adapters cụ thể & Config)
│   ├── __init__.py
│   ├── config.py                # AppConfig & tải .env
│   └── adapters/
│       ├── __init__.py
│       ├── dummy.py             # Dummy Adapter (offline & unit test)
│       ├── google_ai_studio.py  # Google AI Studio Adapter (text-embedding-004)
│       ├── ollama.py            # Ollama Adapter (nomic-embed-text, bge-m3, ...)
│       └── huggingface.py       # Hugging Face Adapter (Inference API & Local)
├── presentation/                # Tầng Presentation (Formatters & Schemas)
│   ├── __init__.py
│   ├── cli/
│   │   ├── __init__.py
│   │   └── handlers.py          # Format kết quả ra terminal
│   └── api/
│       ├── __init__.py
│       └── schemas.py           # Request / Response Schemas cho Web API
└── entrypoints/                 # Thư mục khởi chạy dự án
    ├── __init__.py
    ├── cli.py                   # Điểm chạy CLI
    └── api.py                   # Điểm chạy HTTP Server
```

---

## 🚀 Hướng dẫn chạy dự án

### 1. Chạy CLI

---

## ⚙️ Cấu hình môi trường (Environment Variables)

Đặt các biến môi trường cần thiết trong file `.env` (hoặc trực tiếp trong môi trường hệ thống). Đây là các **key** mà hệ thống sẽ đọc; giá trị có thể để trống nếu không cần thiết cho provider bạn đang dùng.

```dotenv
# Provider mặc định (dummy, google_ai_studio, ollama, huggingface)
EMBEDDING_PROVIDER=

# Google AI Studio
GEMINI_API_KEY=
GOOGLE_EMBEDDING_MODEL=

# Ollama
OLLAMA_BASE_URL=
OLLAMA_MODEL=

# Hugging Face
HF_TOKEN=
HF_MODEL=
HF_USE_LOCAL=false

# Qdrant Vector Database
QDRANT_URL=
QDRANT_API_KEY=

# SQL Database (SQLite mặc định)
SQL_DB_PATH=
SQL_DATABASE_URL=
```

> **Lưu ý**: Các biến trên chỉ được **đọc**; không có giá trị mặc định nào được cung cấp trong mã nguồn. Bạn cần cung cấp chúng trong `.env` hoặc khai báo trực tiếp trong môi trường hệ thống trước khi chạy.

### 2. Chạy HTTP API Server


#### Sử dụng Dummy Adapter (mặc định không cần API key):
```bash
python -m src.entrypoints.cli --text "Kiến trúc Clean Architecture cho Text Embedding" --provider dummy
```

#### Nhúng nhiều văn bản (Batch):
```bash
python -m src.entrypoints.cli --batch "Câu số một" "Câu số hai" "Câu số ba" --provider dummy
```

#### Sử dụng Google AI Studio:
```bash
# Cấu hình API key trong .env hoặc truyền biến môi trường:
export GEMINI_API_KEY="your_api_key"
python -m src.entrypoints.cli --text "Nhúng với Google AI Studio" --provider google_ai_studio
```

#### Sử dụng Ollama:
```bash
# Đảm bảo Ollama server đang chạy: ollama run nomic-embed-text
python -m src.entrypoints.cli --text "Nhúng với Ollama" --provider ollama --model nomic-embed-text
```

#### Sử dụng Hugging Face:
```bash
export HF_TOKEN="your_hf_token"
python -m src.entrypoints.cli --text "Nhúng với Hugging Face" --provider huggingface --model "sentence-transformers/all-MiniLM-L6-v2"
```

> **Chế độ offline**: Đặt `HF_USE_LOCAL=true` trong file `.env` (hoặc môi trường) và cài `sentence-transformers`. Khi bật, adapter sẽ tải mô hình về cache và thực thi hoàn toàn cục bộ, không yêu cầu kết nối mạng.

```bash
export HF_TOKEN="your_hf_token"
python -m src.entrypoints.cli --text "Nhúng với Hugging Face" --provider huggingface --model "sentence-transformers/all-MiniLM-L6-v2"
```

---

### 2. Chạy HTTP API Server

Khởi động server trên cổng 8000:
```bash
python -m src.entrypoints.api --port 8000 --provider dummy
```

Kiểm tra endpoint:
- **Health check**:
  ```bash
  curl http://localhost:8000/health
  ```
- **Nhúng 1 văn bản**:
  ```bash
  curl -X POST http://localhost:8000/api/v1/embed \
       -H "Content-Type: application/json" \
       -d '{"text": "Xin chào thế giới"}'
  ```
- **Nhúng danh sách văn bản**:
  ```bash
  curl -X POST http://localhost:8000/api/v1/batch-embed \
       -H "Content-Type: application/json" \
       -d '{"texts": ["Đoạn văn 1", "Đoạn văn 2"]}'
  ```
