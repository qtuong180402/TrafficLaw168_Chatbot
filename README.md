# Traffic Law Chatbot (Decree 168) 🚗👮‍♂️

Chatbot hỏi đáp về Luật Giao thông (Nghị định 168/2024/NĐ-CP) sử dụng mô hình ngôn ngữ lớn (LLM) chạy local với RAG (Retrieval-Augmented Generation).

## 🛠️ Yêu cầu hệ thống

1.  **Python 3.8+**
2.  **Ollama**: Để chạy LLM local.

## 🚀 Cài đặt

### Bước 1: Cài đặt và cấu hình Ollama

1.  Tải và cài đặt Ollama tại: [https://ollama.com/](https://ollama.com/)
2.  Mở terminal (CMD/PowerShell) và kéo model `llama3.2` (hoặc model khác bạn muốn dùng):
    ```bash
    ollama pull llama3.2
    ```
3.  Đảm bảo Ollama đang chạy (thường nó sẽ chạy ngầm ở background, icon dưới thanh taskbar).

### Bước 2: Cài đặt thư viện Python

Tại thư mục dự án, chạy lệnh sau để cài các thư viện cần thiết:

```bash
pip install -r requirements.txt
```

*Lưu ý: Nếu gặp lỗi với thư viện `torch` hoặc `faiss`, hãy đảm bảo bạn đã cài đặt Python phiên bản tương thích và có C++ build tools nếu cần.*

## ▶️ Chạy ứng dụng

1.  Mở terminal tại thư mục dự án.
2.  Chạy ứng dụng bằng Streamlit:

```bash
streamlit run app.py
```

3.  Trình duyệt sẽ tự động mở địa chỉ `http://localhost:8501`.

## 📖 Hướng dẫn sử dụng

1.  **Lần đầu chạy**: Nhấn nút **"🔄 Build Knowledge Base (RAG)"** để hệ thống đọc file tài liệu và tạo dữ liệu tìm kiếm (vector database). Quá trình này có thể mất vài phút tùy vào độ dài tài liệu.
2.  Nhập câu hỏi của bạn vào ô chat (ví dụ: "Vượt đèn đỏ bị phạt bao nhiêu?").
3.  Chatbot sẽ tìm kiếm thông tin trong nghị định và trả lời bạn.

## 📂 Cấu trúc thư mục

*   `app.py`: Giao diện chính (Streamlit).
*   `services/`: Chứa logic xử lý.
    *   `ollama_client.py`: Kết nối với Ollama.
    *   `rag_engine.py`: Xử lý đọc tài liệu và tìm kiếm (RAG).
*   `data/docs/`: Chứa tài liệu luật (PDF, Word, TXT). Bạn có thể thêm file luật mới vào đây.
*   `models/`: Chứa prompt mẫu.
