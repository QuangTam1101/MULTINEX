"""
=============================================================================
SCRIPT NẠP DỮ LIỆU ĐA NGUỒN (DATASET + WEBSITE)
=============================================================================
"""
import os
import json
import logging
import argparse
import time
import requests
from bs4 import BeautifulSoup
from typing import List, Dict
from dotenv import load_dotenv

load_dotenv()

import chromadb
from chromadb import Documents, EmbeddingFunction, Embeddings
from sentence_transformers import SentenceTransformer

# =============================================================================
# 1. CẤU HÌNH DANH SÁCH WEBSITE ĐÃ KIỂM CHỨNG
# =============================================================================
# Bạn hãy điền các link bài báo uy tín vào đây
VERIFIED_URLS = [
    "https://nhandan.vn/dan-toc-tay-post723931.html",
    "https://sodantoctongiao.hanoi.gov.vn/articles/3380",
    "https://vi.wikipedia.org/wiki/Ng%C6%B0%E1%BB%9Di_T%C3%A0y",
    "https://baophapluat.vn/dac-sac-van-hoa-truyen-thong-cua-nguoi-bahnar-hre-o-an-lao-post481297.html",
    "https://vi.wikipedia.org/wiki/Ng%C6%B0%E1%BB%9Di_Ba_Na"
]

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger(__name__)

COLLECTION_NAME = "tay_bana_culture"
CHROMA_PATH = "./chroma_db"

# =============================================================================
# 2. CLASS EMBEDDING LOCAL (Giữ nguyên)
# =============================================================================
class LocalEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        print("⏳ [Embedding] Đang tải model (lần đầu mất vài giây)...")
        self.model = SentenceTransformer('keepitreal/vietnamese-sbert')
        print("✅ [Embedding] Đã tải xong model!")

    def __call__(self, input: Documents) -> Embeddings:
        embeddings = self.model.encode(input)
        return embeddings.tolist()
    
    def name(self) -> str:
        return "local_vietnamese_sbert"

# =============================================================================
# 3. HÀM XỬ LÝ DATASET JSONL (Nguồn 2)
# =============================================================================
def process_jsonl(file_path: str) -> List[Dict]:
    data_items = []
    print(f"📂 Đang đọc Dataset: {file_path}...")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                try:
                    obj = json.loads(line)
                    messages = obj.get('messages', [])
                    q = next((m['content'] for m in messages if m['role'] == 'user'), "")
                    a = next((m['content'] for m in messages if m['role'] == 'assistant'), "")
                    if q and a:
                        data_items.append({
                            "id": f"jsonl_{i}",
                            "content": f"Câu hỏi: {q}\nTrả lời: {a}",
                            "metadata": {
                                "source": "Dataset Nội Bộ", # Nguồn dataset
                                "type": "qa"
                            }
                        })
                except json.JSONDecodeError:
                    continue
    except Exception as e:
        logger.error(f"❌ Lỗi đọc file JSONL: {e}")
    return data_items

# =============================================================================
# 4. HÀM CRAWL WEBSITE & CẮT NHỎ VĂN BẢN (Nguồn 1)
# =============================================================================
def crawl_website(url: str) -> List[Dict]:
    """Đọc nội dung từ URL và cắt nhỏ thành các đoạn"""
    print(f"🌐 Đang cào dữ liệu từ: {url} ...")
    chunks = []
    try:
        # Giả lập trình duyệt để không bị chặn
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'}
        response = requests.get(url, headers=headers, timeout=10)
        
        if response.status_code != 200:
            logger.error(f"   ⚠️ Không truy cập được {url} (Code {response.status_code})")
            return []

        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Lấy tiêu đề bài viết
        title = soup.title.string if soup.title else url
        
        # Lấy toàn bộ văn bản (đã loại bỏ thẻ HTML)
        # Tùy website mà bạn có thể cần chỉnh 'p' thành 'article' hoặc div cụ thể
        paragraphs = soup.find_all('p')
        full_text = "\n".join([p.get_text() for p in paragraphs if len(p.get_text()) > 50])
        
        # CẮT NHỎ (CHUNKING): Mỗi đoạn khoảng 500 ký tự để Bot dễ đọc
        chunk_size = 500
        overlap = 50
        
        text_len = len(full_text)
        for i in range(0, text_len, chunk_size - overlap):
            chunk_text = full_text[i : i + chunk_size]
            if len(chunk_text) < 100: continue # Bỏ qua đoạn quá ngắn
            
            chunks.append({
                "id": f"web_{hash(url)}_{i}",
                "content": f"Nguồn: {title}\nNội dung: {chunk_text}",
                "metadata": {
                    "source": url,       # Lưu Link để trích dẫn
                    "title": title,      # Lưu tiêu đề
                    "type": "website"
                }
            })
            
        print(f"   ✅ Đã lấy được {len(chunks)} đoạn thông tin từ bài viết này.")
        
    except Exception as e:
        logger.error(f"   ❌ Lỗi khi cào {url}: {e}")
        
    return chunks

# =============================================================================
# 5. HÀM TẠO DB CHÍNH
# =============================================================================
def create_db(input_file: str, recreate: bool = False):
    chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
    ef = LocalEmbeddingFunction()

    if recreate:
        try:
            chroma_client.delete_collection(COLLECTION_NAME)
            print("🗑️  Đã xóa dữ liệu cũ.")
        except:
            pass

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME, embedding_function=ef)
    
    all_items = []
    
    # 1. Xử lý Dataset
    all_items.extend(process_jsonl(input_file))
    
    # 2. Xử lý Website
    for url in VERIFIED_URLS:
        all_items.extend(crawl_website(url))

    if not all_items:
        print("❌ Không có dữ liệu nào được thu thập!")
        return

    # Nạp vào DB
    batch_size = 50
    total = len(all_items)
    print(f"\n🚀 Đang nạp tổng cộng {total} mục kiến thức vào não Bot...")
    
    for i in range(0, total, batch_size):
        batch = all_items[i:i+batch_size]
        try:
            collection.add(
                ids=[item['id'] for item in batch],
                documents=[item['content'] for item in batch],
                metadatas=[item['metadata'] for item in batch]
            )
            print(f"   ...Đã nạp {min(i+batch_size, total)}/{total}")
        except Exception as e:
            logger.error(f"Lỗi batch {i}: {e}")

    print("\n🎉 HOÀN TẤT NÂNG CẤP DỮ LIỆU!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="sample_data.jsonl")
    parser.add_argument("--recreate", action="store_true")
    args = parser.parse_args()
    
    create_db(args.input, args.recreate or True)