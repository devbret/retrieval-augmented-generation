import os, requests, json
from typing import List, Optional
from fastapi import FastAPI, Body
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "local_rag")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai").lower()

RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "") 

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "mistral-8x7b-instruct")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:instruct")

client = chromadb.PersistentClient(path=CHROMA_DIR, settings=Settings(allow_reset=False))
collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space":"cosine"})
embedder = SentenceTransformer(EMBED_MODEL)

reranker = None
if RERANK_MODEL_NAME:
    try:
        print(f"[reranker] Loading CrossEncoder: {RERANK_MODEL_NAME}")
        reranker = CrossEncoder(RERANK_MODEL_NAME)
    except Exception as e:
        print(f"[reranker] Could not load reranker '{RERANK_MODEL_NAME}': {e}")
        reranker = None

app = FastAPI(title="Local RAG (Mistral)")

class AskReq(BaseModel):
    question: str
    k: int = 5
    max_tokens: int = 512
    temperature: float = 0.2

    min_score: float = 0.0               
    sources: Optional[List[str]] = None 

    answer_style: str = "default"

class AskResp(BaseModel):
    answer: str
    sources: List[dict]

SYSTEM = (
    "You are a concise assistant. Use ONLY the provided context to answer. "
    "If the answer is not in the context, say you don't know. Cite sources as [n]."
)

def build_prompt(question: str, docs: List[str], style: str = "default") -> str:
    ctx_lines = []
    for i, d in enumerate(docs, 1):
        ctx_lines.append(f"[{i}] {d}")
    ctx = "\n\n".join(ctx_lines)

    extra_style = ""
    if style == "bullets":
        extra_style = (
            "\nFormat the answer as concise bullet points. "
            "Always include citations like [1], [2] next to the bullet(s) they come from."
        )
    elif style == "json":
        extra_style = (
            "\nReturn your answer as JSON with keys: "
            "`answer` (string), `bullets` (array of short bullet strings), "
            "`citations` (array of integers referencing the context numbers). "
            "Do not include any additional text outside the JSON."
        )

    return (
        f"Context:\n{ctx}\n\n"
        f"Question: {question}\n\n"
        f"Answer (with citations like [1], [2]):{extra_style}"
    )

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
    chat_payload = {
        "model": OLLAMA_MODEL,
        "options": {"temperature": temperature, "num_predict": max_tokens},
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
        "stream": False,
    }

    chat_url = f"{OLLAMA_BASE_URL}/api/chat"
    r = requests.post(chat_url, json=chat_payload, timeout=120)

    if r.status_code == 404:
        gen_url = f"{OLLAMA_BASE_URL}/api/generate"
        gen_payload = {
            "model": OLLAMA_MODEL,
            "prompt": f"{SYSTEM}\n\n{prompt}",
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        r = requests.post(gen_url, json=gen_payload, timeout=120)
        r.raise_for_status()
        data = r.json()
        if "response" in data:
            return data["response"]
        return json.dumps(data)

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

@app.get("/")
def health():
    return {"ok": True, "collection": COLLECTION_NAME}

@app.post("/ask", response_model=AskResp)
def ask(req: AskReq = Body(...)):
    q_emb = embedder.encode([req.question])[0].tolist()

    where = {}
    if req.sources:
        where = {"source": {"$in": req.sources}}

    res = collection.query(
        query_embeddings=[q_emb],
        n_results=req.k,
        include=["documents","metadatas","distances"],
        where=where or None,
    )

    docs = res["documents"][0] if res["documents"] else []
    metas = res["metadatas"][0] if res["metadatas"] else []
    dists = res["distances"][0] if res["distances"] else []

    if not docs:
        return AskResp(
            answer="I don't have any context yet, or no documents matched your filters. Please ingest documents.",
            sources=[]
        )

    rerank_scores = [None] * len(docs)
    if reranker is not None and docs:
        pairs = [(req.question, d) for d in docs]
        scores = reranker.predict(pairs).tolist()
        ranked = sorted(
            zip(docs, metas, dists, scores),
            key=lambda x: x[3],
            reverse=True
        )
        docs, metas, dists, rerank_scores = zip(*ranked)
        docs = list(docs)
        metas = list(metas)
        dists = list(dists)
        rerank_scores = list(rerank_scores)

    filtered_docs = []
    filtered_metas = []
    filtered_dists = []
    filtered_rr = []

    for d, m, dist, rr in zip(docs, metas, dists, rerank_scores):
        score = 1.0 - float(dist)
        if score >= req.min_score:
            filtered_docs.append(d)
            filtered_metas.append(m)
            filtered_dists.append(dist)
            filtered_rr.append(rr)

    if not filtered_docs:
        return AskResp(
            answer="I found only weak matches in the database based on your min_score threshold.",
            sources=[]
        )

    prompt = build_prompt(req.question, filtered_docs, style=req.answer_style)
    answer = llm_answer(prompt, max_tokens=req.max_tokens, temperature=req.temperature)

    sources = []
    for i, (m, d, rr) in enumerate(zip(filtered_metas, filtered_dists, filtered_rr), 1):
        sources.append({
            "id": i,
            "source": m.get("source","unknown"),
            "chunk": m.get("chunk"),
            "score": 1.0 - float(d),
            "rerank_score": rr,
        })

    return AskResp(answer=answer, sources=sources)
