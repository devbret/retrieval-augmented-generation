import os, requests, json
from typing import List
from fastapi import FastAPI, Body
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

# --- ENV ---
CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "local_rag")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai").lower()

# OpenAI-like
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "mistral-8x7b-instruct")

# Ollama
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:instruct")

# --- Init ---
client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))
collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space":"cosine"})
embedder = SentenceTransformer(EMBED_MODEL)

app = FastAPI(title="Local RAG (Mistral)")

class AskReq(BaseModel):
    question: str
    k: int = 5
    max_tokens: int = 512
    temperature: float = 0.2

class AskResp(BaseModel):
    answer: str
    sources: List[dict]

# --- Prompt Template ---
SYSTEM = (
    "You are a concise assistant. Use ONLY the provided context to answer. "
    "If the answer is not in the context, say you don't know. Cite sources as [n]."
)

def build_prompt(question: str, docs: List[str]) -> str:
    ctx_lines = []
    for i, d in enumerate(docs, 1):
        ctx_lines.append(f"[{i}] {d}")
    ctx = "\n\n".join(ctx_lines)
    return f"Context:\n{ctx}\n\nQuestion: {question}\n\nAnswer (with citations like [1], [2]):"

# --- LLM Callers ---
def call_openai_like(prompt: str, max_tokens: int, temperature: float) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"} if OPENAI_API_KEY else {}
    payload = {
        "model": OPENAI_MODEL,
        "messages": [
            {"role":"system", "content": SYSTEM},
            {"role":"user", "content": prompt},
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }
    r = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def call_ollama(prompt: str, max_tokens: int, temperature: float) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "messages": [
            {"role":"system", "content": SYSTEM},
            {"role":"user", "content": prompt},
        ],
        "stream": False
    }
    r = requests.post(f"{OLLAMA_BASE_URL}/api/chat", json=payload, timeout=120)
    r.raise_for_status()
    data = r.json()
    if "message" in data and "content" in data["message"]:
        return data["message"]["content"]
    if "choices" in data:
        return data["choices"][0]["message"]["content"]
    return json.dumps(data)

def llm_answer(prompt: str, max_tokens: int, temperature: float) -> str:
    if LLM_BACKEND == "ollama":
        return call_ollama(prompt, max_tokens, temperature)
    return call_openai_like(prompt, max_tokens, temperature)

# --- Routes ---
@app.get("/")
def health():
    return {"ok": True, "collection": COLLECTION_NAME}

@app.post("/ask", response_model=AskResp)
def ask(req: AskReq = Body(...)):
    q_emb = embedder.encode([req.question])[0].tolist()
    res = collection.query(query_embeddings=[q_emb], n_results=req.k, include=["documents","metadatas","distances"])
    docs = res["documents"][0] if res["documents"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []
    dists = res["distances"][0] if res["distances"] else []

    if not docs:
        return AskResp(answer="I don't have any context yet. Please ingest documents.", sources=[])

    prompt = build_prompt(req.question, docs)
    answer = llm_answer(prompt, max_tokens=req.max_tokens, temperature=req.temperature)

    sources = []
    for i, (m, d) in enumerate(zip(metas, dists), 1):
        sources.append({"id": i, "source": m.get("source","unknown"), "chunk": m.get("chunk"), "score": 1.0 - float(d)})

    return AskResp(answer=answer, sources=sources)
