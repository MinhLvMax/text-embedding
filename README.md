# text-embedding

Công cụ phụ trợ trong pipeline xử lý dữ liệu (chuẩn bị dữ liệu cho RAG / đồ thị tri thức):

- **Đầu vào:** Đọc các đoạn text chunk từ file SQLite nguồn (`staging_chunks.db`).
- **Xử lý:** Chọn và chạy mô hình để tạo vector embedding (hỗ trợ Hugging Face local, Ollama, Gemini).
- **Đầu ra:** Lưu vector vào SQLite (để sao lưu) và đẩy vào Qdrant (để tìm kiếm ngữ nghĩa).
