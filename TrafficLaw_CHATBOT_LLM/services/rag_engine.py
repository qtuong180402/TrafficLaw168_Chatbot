import os
import pickle
import faiss
import re
from pypdf import PdfReader
from docx import Document
from sentence_transformers import SentenceTransformer
import numpy as np

class RAGEngine:
    def __init__(self, docs_path="data/docs", vec_path="data/vector_store.faiss", chunk_cache="data/chunks.pkl"):
        self.docs_path = docs_path
        self.vec_path = vec_path
        self.chunk_cache = chunk_cache

        self.model = SentenceTransformer("AITeamVN/Vietnamese_Embedding")

        self.chunks = []
        self.index = None
        
        if os.path.exists(self.chunk_cache) == False or os.path.exists(self.vec_path) == False:
            print("RAG resources not found, please build the knowledge base first.")
            self.build()
        else:
            self.load_chunks()
            self.load_index()

    # Load text from PDF
    def load_pdf(self, filepath):
        reader = PdfReader(filepath)
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text

    # Load text from DOCX
    def load_docx(self, filepath):
        doc = Document(filepath)
        return "\n".join([p.text for p in doc.paragraphs])

    # Extract all documents
    def load_all_docs(self):
        texts = []

        for filename in os.listdir(self.docs_path):
            fullpath = os.path.join(self.docs_path, filename)

            if filename.endswith(".pdf"):
                texts.append(self.load_pdf(fullpath))
            elif filename.endswith(".docx"):
                texts.append(self.load_docx(fullpath))
            elif filename.endswith(".txt"):
                with open(fullpath, "r", encoding="utf-8") as f:
                    texts.append(f.read())

        return texts

    # Chunking: Structural (theo Điều) + Recursive (theo đoạn/từ)
    def chunk_text(self, text, size=400, overlap=50):
        # Bước 1: Structural Chunking - Tách theo "Điều"
        # Regex tìm "Điều <số>." ở đầu dòng
        article_pattern = r'(?=\nĐiều \d+\.)'
        
        # Thêm \n vào đầu text để đảm bảo bắt được Điều đầu tiên
        articles = re.split(article_pattern, '\n' + text)
        articles = [a.strip() for a in articles if a.strip()]
        
        final_chunks = []
        
        for article in articles:
            words = article.split()
            
            # Bước 2: Recursive-like logic
            # Nếu chunk Điều đủ nhỏ, giữ nguyên (đây là ngữ cảnh tốt nhất)
            if len(words) <= size:
                final_chunks.append(article)
            else:
                # Nếu chunk quá dài, tách nhỏ hơn theo thứ tự ưu tiên: Dòng (Khoản) -> Sliding Window
                lines = article.split('\n')
                current_chunk = []
                current_length = 0
                
                header = lines[0] # Giữ lại dòng tiêu đề "Điều X..." để ghép vào các chunk con (Contextual preservation)
                
                for i, line in enumerate(lines):
                    line_words = line.split()
                    line_len = len(line_words)
                    
                    # Nếu 1 dòng (Khoản) quá dài -> Dùng Sliding Window cắt nhỏ dòng đó
                    if line_len > size:
                        # Trước hết, lưu chunk đang tích lũy (nếu có)
                        if current_chunk:
                            chunk_text = "\n".join(current_chunk)
                            if header not in chunk_text and i > 0: # Thêm ngữ cảnh Điều nếu thiếu
                                chunk_text = header + "\n...\n" + chunk_text
                            final_chunks.append(chunk_text)
                            current_chunk = []
                            current_length = 0
                        
                        # Cắt dòng dài này
                        step = size - overlap
                        if step <= 0: step = 1
                        for j in range(0, line_len, step):
                            sub_words = line_words[j:j + size]
                            sub_text = " ".join(sub_words)
                            if header not in sub_text and i > 0:
                                sub_text = header + " (tiếp)\n...\n" + sub_text
                            final_chunks.append(sub_text)
                            
                    # Nếu cộng dồn vượt quá size -> Lưu chunk và reset
                    elif current_length + line_len > size:
                        chunk_text = "\n".join(current_chunk)
                        if header not in chunk_text and i > 0:
                            chunk_text = header + "\n...\n" + chunk_text
                        final_chunks.append(chunk_text)
                        
                        # Reset chunk mới (có thể bắt đầu bằng overlap dòng trước - ở đây làm đơn giản là bắt đầu dòng mới)
                        current_chunk = [line]
                        current_length = line_len
                    else:
                        current_chunk.append(line)
                        current_length += line_len
                
                # Lưu phần dư cuối cùng
                if current_chunk:
                    chunk_text = "\n".join(current_chunk)
                    if header not in chunk_text:
                        chunk_text = header + "\n...\n" + chunk_text
                    final_chunks.append(chunk_text)

        return final_chunks

    # Build vector store
    def build(self):
        texts = self.load_all_docs()
        for t in texts:
            self.chunks.extend(self.chunk_text(t))

        # Save chunk cache
        with open(self.chunk_cache, "wb") as f:
            pickle.dump(self.chunks, f)

        # Encode chunks
        vectors = self.model.encode(self.chunks)
        vectors = np.array(vectors).astype("float32")

        # FAISS index
        dim = vectors.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(vectors)

        # Save FAISS index
        faiss.write_index(self.index, self.vec_path)

    def load_chunks(self):
        if os.path.exists(self.chunk_cache):
            with open(self.chunk_cache, "rb") as f:
                self.chunks = pickle.load(f)

    def load_index(self):
        if os.path.exists(self.vec_path):
            self.index = faiss.read_index(self.vec_path)

    # Query top-k documents
    def search(self, query, k=3):
        if self.index is None:
            return []

        q_vec = self.model.encode([query]).astype("float32")
        D, I = self.index.search(q_vec, k)

        return [self.chunks[i] for i in I[0]]