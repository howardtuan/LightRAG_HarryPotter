import asyncio
import os
import inspect
import logging
from lightrag import LightRAG, QueryParam
from lightrag.llm.ollama import ollama_model_complete, ollama_embed
from lightrag.utils import EmbeddingFunc

WORKING_DIR = "./dickens"

logging.basicConfig(format="%(levelname)s:%(message)s", level=logging.INFO)

if not os.path.exists(WORKING_DIR):
    os.mkdir(WORKING_DIR)

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

with open("./book.txt", "r", encoding="utf-8") as f:
    rag.insert(f.read())
print('====== naive response START======')

# Perform naive search
print(
    rag.query("What is the meaning of you-know-who?", param=QueryParam(mode="naive"))
)

print('====== naive response END======')
print('................................................')
print('====== local response START======')

# Perform local search
print(
    rag.query("What is the meaning of you-know-who?", param=QueryParam(mode="local"))
)
print('====== local response END======')
print('................................................')
print('====== global response START======')

# Perform global search
print(
    rag.query("What is the meaning of you-know-who?", param=QueryParam(mode="global"))
)

print('====== global response END======')
print('................................................')
print('====== hybrid response START======')

# Perform hybrid search
print(
    rag.query("What is the meaning of you-know-who?", param=QueryParam(mode="hybrid"))
)

print('====== hybrid response END======')
print('................................................')


# stream response

# resp = rag.query(
#     "What are the top themes in this story?",
#     param=QueryParam(mode="hybrid", stream=True),
# )


# async def print_stream(stream):
#     async for chunk in stream:
#         print(chunk, end="", flush=True)


# if inspect.isasyncgen(resp):
#     asyncio.run(print_stream(resp))
# else:
#     print(resp)
