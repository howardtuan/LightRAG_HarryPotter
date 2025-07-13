import os
import logging
from flask import Flask, request, jsonify, render_template
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

# Flask 應用初始化
app = Flask(__name__)

# 配置 Logger
logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

# 設定工作目錄
WORKING_DIR = "./dickens"
if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

# 在 Flask 啟動時，就初始化 LightRAG
rag = LightRAG(
    working_dir=WORKING_DIR,
    llm_model_func=ollama_model_complete,
    llm_model_name="llama3.1",
    llm_model_max_async=4,
    llm_model_max_token_size=32768,
    llm_model_kwargs={"host": "http://localhost:11434", "options": {"num_ctx": 32768}},
    embedding_func=EmbeddingFunc(
        embedding_dim=768,
        max_token_size=8192,
        func=lambda texts: ollama_embed(
            texts, embed_model="nomic-embed-text", host="http://localhost:11434"
        ),
    ),
)

# 讀取 book.txt ，並插入到資料庫（僅需在啟動時做一次）
BOOK_PATH = "./book.txt"
if os.path.exists(BOOK_PATH):
    with open(BOOK_PATH, "r", encoding="utf-8") as f:
        content = f.read()
        rag.insert(content)
        logging.info("Document inserted into LightRAG.")
else:
    logging.warning("book.txt does not exist. Please put your text file in the specified path.")

@app.route("/")
def index():
    """
    回傳前端主頁
    """
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    """
    接收前端問題並回傳 RAG 的回答
    """
    data = request.json  # 前端送 JSON
    user_query = data.get("query", "")
    mode = data.get("mode", "naive")

    # 使用指定的 mode 進行搜尋
    response = rag.query(user_query, param=QueryParam(mode=mode))

    # 回傳 JSON
    return jsonify({"answer": response})

if __name__ == "__main__":
    # 啟動 Flask 伺服器
    # debug=True 方便開發時使用，正式部署請關閉
    app.run(host="0.0.0.0", port=5000, debug=True)
