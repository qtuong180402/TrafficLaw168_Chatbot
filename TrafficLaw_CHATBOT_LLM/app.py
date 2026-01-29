import streamlit as st
from services.ollama_client import OllamaClient
from services.rag_engine import RAGEngine

# Streamlit config
st.set_page_config(page_title="Traffic Laws Decree 168 Chatbot (Local LLM)", page_icon="📜")

SYSTEM_PROMPT = """Bạn là một trợ lý AI thông minh, chuyên gia về Luật Giao thông Đường bộ Việt Nam, đặc biệt là Nghị định 168/2024/NĐ-CP.

Nguyên tắc bắt buộc:
1. Chỉ sử dụng thông tin có trong các đoạn văn bản được cung cấp (context).
2. Không được suy luận, không bổ sung kiến thức bên ngoài Nghị định 168.
3. Nếu không tìm thấy quy định phù hợp, phải trả lời rõ: 
   "Nghị định 168 không quy định cụ thể trường hợp này."
4. Khi trả lời về mức phạt, phải nêu rõ:
   - Trích dẫn từ văn bản theo cấu trúc "Chương > Mục > Điều > Khoản > Điểm":
        + Chương (ví dụ Chương I, Chương II, v.v.)
        + Mục (nếu có, ví dụ Mục 1, Mục 2, v.v.)
        + Điều (ví dụ: Điều 1, Điều 2, v.v.) 
        + Khoản (ví dụ 1, 2, 3, v.v.)
        + Điểm (ví dụ a, b, c, v.v.)
   - Đối tượng áp dụng (loại phương tiện)
   - Mức phạt chính xác
5. Không đưa ra lời khuyên cá nhân hay đánh giá chủ quan.
6. Ngôn ngữ rõ ràng, trung lập, dễ hiểu.

Cách trả lời:
- Ưu tiên liệt kê theo gạch đầu dòng.
- Trích dẫn điều khoản theo dạng: 
  "Theo Điều X, Khoản Y Nghị định 168..."
- Không sử dụng các cụm từ phỏng đoán như "có thể", "thường là", "nhiều khả năng".

Nếu câu hỏi không rõ thông tin (ví dụ: loại phương tiện, hành vi cụ thể),
hãy yêu cầu người hỏi cung cấp thêm thông tin cần thiết trước khi trả lời.

"""

# Init client
@st.cache_resource
def get_resources():
    client = OllamaClient(model_name="llama3.2")
    rag = RAGEngine()
    return client, rag
    
client, rag = get_resources()

st.title("🚦 Traffic Laws Decree 168 Q&A Chatbot")
st.write("Powered by Local Ollama3.2 + RAG")

# Build RAG button
if st.button("🔄 Build Knowledge Base (RAG)"):
    rag.build()
    st.success("✅ Knowledge base built successfully!")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display old messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# User input
user_input = st.chat_input("🚦 Nhập câu hỏi của bạn về luật giao thông (nghị định 168)...")

if user_input:
    # Save user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)

    # Model reply with RAG
    with st.chat_message("assistant"):

        # RAG augmentation
        related_chunks = rag.search(user_input)
        context = "\n\n".join(related_chunks)
        print("RAG Context:", context)
        with st.expander("🐞 RAG Retrieved Context (debug)"):
            st.write(context)

        full_prompt = f"""
Dưới đây là một số thông tin từ tài liệu luật giao thông (nghị định 168):

{context}

Dựa trên thông tin này, trả lời câu hỏi của người dùng:
{user_input}
"""

        reply = client.ask(SYSTEM_PROMPT, full_prompt)
        st.write(reply)

    # Save assistant message
    st.session_state.messages.append({"role": "assistant", "content": reply})