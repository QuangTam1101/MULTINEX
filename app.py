"""
=============================================================================
HỆ THỐNG RAG CHATBOT (DATASET + WEB)
=============================================================================
"""
import os
import time
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
load_dotenv()

from google import genai
from google.genai import types
import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY", "")
gemini_client = None
if GOOGLE_API_KEY:
    gemini_client = genai.Client(api_key=GOOGLE_API_KEY)

chroma_client = chromadb.PersistentClient(path="./chroma_db")
COLLECTION_NAME = "tay_bana_culture"

# --- EMBEDDING ---
class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        self.model = SentenceTransformer('keepitreal/vietnamese-sbert')
    def __call__(self, input: Documents) -> Embeddings:
        return self.model.encode(input).tolist()
    def name(self) -> str:
        return "local_vietnamese_sbert"

print("\n⏳ Đang tải Model AI...")
GLOBAL_EMBEDDING_FUNC = LocalEmbeddingFunction()
print("✅ Server sẵn sàng!\n")

def get_collection():
    return chroma_client.get_or_create_collection(
        name=COLLECTION_NAME, embedding_function=GLOBAL_EMBEDDING_FUNC
    )

def retry_api_call(func, *args, **kwargs):
    for attempt in range(3):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            if "429" in str(e) and attempt < 2:
                time.sleep(2 * (2 ** attempt))
                continue
            raise e

@app.route('/api/chat', methods=['POST'])
def chat():
    if not gemini_client: return jsonify({"error": "No API Key"}), 500
    
    data = request.json
    user_query = data.get('message', '')
    if not user_query: return jsonify({"error": "No query"}), 400

    try:
        # 1. Tìm kiếm
        collection = get_collection()
        results = collection.query(query_texts=[user_query], n_results=5)
        
        retrieved_docs = []
        unique_sources = {} # Dùng để lọc nguồn trùng lặp

        if results and results['documents']:
            for i, doc in enumerate(results['documents'][0]):
                retrieved_docs.append(doc)
                
                # Lấy Metadata để trích dẫn nguồn
                meta = results['metadatas'][0][i] if results['metadatas'] else {}
                source_url = meta.get("source", "Tài liệu nội bộ")
                source_title = meta.get("title", "Dataset")
                
                # Chỉ lưu nguồn nếu chưa có trong danh sách (tránh trùng)
                if source_url not in unique_sources:
                    unique_sources[source_url] = source_title

        # 2. Tạo Prompt
        context_text = "\n\n---\n\n".join(retrieved_docs)
        
        system_instruction = """Bạn là chuyên gia văn hóa. Trả lời dựa trên thông tin được cung cấp.
        Nếu thông tin đến từ website, hãy nhắc đến tên website đó trong câu trả lời."""

        full_prompt = f"DỮ LIỆU THAM KHẢO:\n{context_text}\n\n---\nCÂU HỎI: {user_query}"

        # 3. Gọi Gemini
        response = retry_api_call(
            gemini_client.models.generate_content,
            model="gemini-2.5-flash",
            config=types.GenerateContentConfig(
                system_instruction=system_instruction, 
                temperature=0.3
            ),
            contents=full_prompt
        )

        # 4. Xử lý phần hiển thị nguồn (Hyperlink)
        # Tạo danh sách nguồn dạng Markdown [Tên nguồn](URL)
        source_list_text = "\n\n**Nguồn tham khảo:**\n"
        has_web_source = False
        
        processed_sources_for_frontend = []
        
        for url, title in unique_sources.items():
            if url.startswith("http"):
                # Nếu là Link Website -> Tạo Hyperlink ngắn gọn
                # Cắt ngắn URL nếu quá dài để hiển thị đẹp hơn
                display_url = url
                if len(display_url) > 50: display_url = display_url[:47] + "..."
                
                source_list_text += f"- [{title}]({url})\n"
                processed_sources_for_frontend.append({"content": f"Website: {title}", "url": url})
                has_web_source = True
            else:
                # Nếu là Dataset
                processed_sources_for_frontend.append({"content": "Dữ liệu nội bộ (Dataset)", "url": None})

        # Nếu có nguồn web, nối thêm vào cuối câu trả lời của Bot
        final_answer = response.text
        if has_web_source:
            final_answer += source_list_text

        return jsonify({
            "success": True,
            "answer": final_answer,
            "sources": processed_sources_for_frontend
        })

    except Exception as e:
        logger.error(f"Lỗi: {e}")
        return jsonify({"success": False, "error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)