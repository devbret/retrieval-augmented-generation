import os, re, requests, json, time, threading, queue
from typing import List, Optional
import numpy as np
from fastapi import FastAPI, Body, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from dotenv import load_dotenv
load_dotenv()

import posthog
posthog.disabled = True
posthog.capture = lambda *args, **kwargs: None

import chromadb
from chromadb.config import Settings
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from sentence_transformers import CrossEncoder

from util_text import load_any, split_text, yield_files
from ingest import file_id, legacy_file_id

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "local_rag")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
LLM_BACKEND = os.getenv("LLM_BACKEND", "openai").lower()

DOCS_DIR = os.getenv("DOCS_DIR", "./docs")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

ALLOWED_EXTS = {".txt", ".md", ".markdown", ".pdf", ".csv", ".json", ".html", ".htm", ".log", ".rst"}

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

RERANK_MODEL_NAME = os.getenv("RERANK_MODEL", "")

OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "mistral-8x7b-instruct")

OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:instruct")
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "50000"))

HISTORY_MAX_TURNS = 8

client = chromadb.PersistentClient(
    path=CHROMA_DIR,
    settings=Settings(allow_reset=False, anonymized_telemetry=False),
)
collection = client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space":"cosine"})
try:
    collection.modify(metadata={"hnsw:search_ef": 128})
except Exception as e:
    print(f"[chroma] note: could not raise search_ef ({e})")
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
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

class AskReq(BaseModel):
    question: str
    k: Optional[int] = None
    max_tokens: int = -1
    temperature: float = 0.2

    min_score: float = 0.0
    sources: Optional[List[str]] = None

    answer_style: str = "default"

    history: Optional[List[dict]] = None

class AskResp(BaseModel):
    answer: str
    sources: List[dict]

class SummarizeReq(BaseModel):
    source: str
    max_tokens: int = -1
    temperature: float = 0.2

SYSTEM = (
    "You are a concise assistant. Use ONLY the provided context and the prior "
    "conversation to answer. If the answer is not in the context, say you don't "
    "know. Cite sources as [n]."
)

SUMMARY_SYSTEM = (
    "You are a careful assistant that summarizes documents faithfully. "
    "Cover the main topics, requirements and key definitions. Do not invent content."
)

NO_CONTEXT_MSG = (
    "I don't have any context yet, or no documents matched your filters. "
    "Please ingest documents."
)
WEAK_MATCH_MSG = "I found only weak matches in the database based on your min_score threshold."

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

def build_messages(prompt: str, history: Optional[List[dict]]) -> List[dict]:
    msgs = [{"role": "system", "content": f"{SYSTEM}\n\n{library_overview()}"}]
    for m in (history or [])[-HISTORY_MAX_TURNS:]:
        role = m.get("role")
        content = m.get("content")
        if role in ("user", "assistant") and content:
            msgs.append({"role": role, "content": str(content)[:4000]})
    msgs.append({"role": "user", "content": prompt})
    return msgs


def call_openai_like(messages: List[dict], max_tokens: int, temperature: float) -> str:
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"} if OPENAI_API_KEY else {}
    payload = {
        "model": OPENAI_MODEL,
        "messages": messages,
        "temperature": temperature,
        "stream": False,
    }
    if max_tokens > 0:
        payload["max_tokens"] = max_tokens
    r = requests.post(f"{OPENAI_BASE_URL}/chat/completions", headers=headers, json=payload, timeout=300)
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]

def _ollama_options(max_tokens: int, temperature: float) -> dict:
    num_predict = max_tokens if max_tokens > 0 else -1
    return {"temperature": temperature, "num_predict": num_predict, "num_ctx": OLLAMA_NUM_CTX}

def call_ollama(messages: List[dict], max_tokens: int, temperature: float) -> str:
    chat_payload = {
        "model": OLLAMA_MODEL,
        "options": _ollama_options(max_tokens, temperature),
        "messages": messages,
        "think": False,
        "stream": False,
    }

    chat_url = f"{OLLAMA_BASE_URL}/api/chat"
    r = requests.post(chat_url, json=chat_payload, timeout=300)

    if r.status_code == 400 and "think" in r.text.lower():
        chat_payload.pop("think", None)
        r = requests.post(chat_url, json=chat_payload, timeout=300)

    if r.status_code == 404:
        gen_url = f"{OLLAMA_BASE_URL}/api/generate"
        gen_payload = {
            "model": OLLAMA_MODEL,
            "prompt": "\n\n".join(m["content"] for m in messages),
            "options": _ollama_options(max_tokens, temperature),
            "stream": False,
        }
        r = requests.post(gen_url, json=gen_payload, timeout=300)
        r.raise_for_status()
        data = r.json()
        if "response" in data:
            return data["response"]
        return json.dumps(data)

    r.raise_for_status()
    data = r.json()

    if "message" in data and "content" in data["message"]:
        content = data["message"]["content"]
        if not content and data["message"].get("thinking"):
            content = data["message"]["thinking"]
        return content
    if "choices" in data:
        return data["choices"][0]["message"]["content"]

    return json.dumps(data)

def stream_ollama(messages: List[dict], max_tokens: int, temperature: float):
    chat_payload = {
        "model": OLLAMA_MODEL,
        "options": _ollama_options(max_tokens, temperature),
        "messages": messages,
        "think": False,
        "stream": True,
    }
    chat_url = f"{OLLAMA_BASE_URL}/api/chat"
    r = requests.post(chat_url, json=chat_payload, stream=True, timeout=300)

    if r.status_code == 400 and "think" in r.text.lower():
        chat_payload.pop("think", None)
        r = requests.post(chat_url, json=chat_payload, stream=True, timeout=300)

    if r.status_code == 404:
        yield call_ollama(messages, max_tokens, temperature)
        return

    r.raise_for_status()
    for line in r.iter_lines():
        if not line:
            continue
        data = json.loads(line)
        tok = data.get("message", {}).get("content", "")
        if tok:
            yield tok
        if data.get("done"):
            break

def llm_answer(messages: List[dict], max_tokens: int, temperature: float) -> str:
    if LLM_BACKEND == "ollama":
        return call_ollama(messages, max_tokens, temperature)
    return call_openai_like(messages, max_tokens, temperature)


_llm_status_cache = {"checked": 0.0, "status": {"reachable": None, "model_available": None}}
_llm_status_lock = threading.Lock()

def check_llm() -> dict:
    if LLM_BACKEND != "ollama":
        return {"reachable": None, "model_available": None}
    try:
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=3)
        r.raise_for_status()
        names = [m.get("name", "") for m in r.json().get("models", [])]
        available = OLLAMA_MODEL in names or any(
            n.split(":")[0] == OLLAMA_MODEL for n in names
        )
        return {"reachable": True, "model_available": available}
    except Exception:
        return {"reachable": False, "model_available": False}

def llm_status(max_age: float = 60.0) -> dict:
    with _llm_status_lock:
        if time.time() - _llm_status_cache["checked"] > max_age:
            _llm_status_cache["status"] = check_llm()
            _llm_status_cache["checked"] = time.time()
        return _llm_status_cache["status"]

_startup_status = llm_status(max_age=0)
if _startup_status["reachable"] is False:
    print(f"[llm] WARNING: cannot reach Ollama at {OLLAMA_BASE_URL}")
elif _startup_status["model_available"] is False:
    print(f"[llm] WARNING: model '{OLLAMA_MODEL}' not found on Ollama at {OLLAMA_BASE_URL}")


def _cosine_scores(q_emb, doc_embs) -> List[float]:
    q = np.asarray(q_emb, dtype=float)
    D = np.asarray(doc_embs, dtype=float)
    qn = q / (np.linalg.norm(q) + 1e-9)
    Dn = D / (np.linalg.norm(D, axis=1, keepdims=True) + 1e-9)
    return (Dn @ qn).tolist()

_bm25 = {"index": None, "ids": [], "docs": [], "metas": []}
_bm25_lock = threading.Lock()

def _bm25_tokenize(text: str) -> List[str]:
    return re.findall(r"[a-z0-9][a-z0-9.\-/]*", text.lower())

def rebuild_bm25():
    res = collection.get(include=["documents", "metadatas"])
    ids, docs, metas = res["ids"], res["documents"], res["metadatas"]
    index = BM25Okapi([_bm25_tokenize(d) for d in docs]) if docs else None
    sources = sorted({m.get("source", "unknown") for m in metas})
    with _bm25_lock:
        _bm25["index"] = index
        _bm25["ids"] = ids
        _bm25["docs"] = docs
        _bm25["metas"] = metas
        _bm25["sources"] = sources

def library_overview() -> str:
    with _bm25_lock:
        sources = list(_bm25.get("sources") or [])
    if not sources:
        return "The knowledge base is currently empty."
    names = [os.path.basename(s) for s in sources]
    shown = "; ".join(names[:150])
    more = f" (and {len(names) - 150} more)" if len(names) > 150 else ""
    return (
        f"The knowledge base contains {len(names)} documents: {shown}{more}. "
        "Only the excerpts most relevant to the current question are provided "
        "as context; the other documents exist and are searchable, they just "
        "were not retrieved for this question."
    )

def bm25_candidates(q_text: str, sources_filter: Optional[set], limit: int) -> List[tuple]:
    with _bm25_lock:
        index = _bm25["index"]
        ids, docs, metas = _bm25["ids"], _bm25["docs"], _bm25["metas"]
    if index is None:
        return []
    scores = index.get_scores(_bm25_tokenize(q_text))
    out = []
    for i in np.argsort(scores)[::-1]:
        if scores[i] <= 0:
            break
        if sources_filter and metas[i].get("source") not in sources_filter:
            continue
        out.append((ids[i], docs[i], metas[i]))
        if len(out) >= limit:
            break
    return out

rebuild_bm25()

def _query_with_fallback(q_emb, n_results: int, where: Optional[dict]):
    n = max(1, n_results)
    while True:
        try:
            return collection.query(
                query_embeddings=[q_emb],
                n_results=n,
                include=["documents", "metadatas", "distances"],
                where=where,
            )
        except RuntimeError as e:
            msg = str(e).lower()
            if ("contigious" in msg or "contiguous" in msg) and n > 1:
                n = n // 2
            else:
                raise

def retrieve(req: AskReq):
    q_text = req.question
    if req.history:
        prev_user = [m.get("content", "") for m in req.history if m.get("role") == "user"]
        if prev_user:
            q_text = f"{prev_user[-1]}\n{req.question}"
    q_emb = embedder.encode([q_text])[0].tolist()

    where = {"source": {"$in": req.sources}} if req.sources else None

    if req.k:
        n_fetch = max(req.k * 4, 20) if reranker is not None else max(req.k * 2, 10)
    else:
        n_fetch = 150
    n_fetch = max(1, min(n_fetch, collection.count()))
    res = _query_with_fallback(q_emb, n_fetch, where)

    vec_ids = res["ids"][0] if res["ids"] else []
    info = {}
    for i, cid in enumerate(vec_ids):
        info[cid] = (res["documents"][0][i], res["metadatas"][0][i], res["distances"][0][i])

    bm_hits = bm25_candidates(
        q_text, set(req.sources) if req.sources else None, limit=n_fetch
    )
    for cid, d, m in bm_hits:
        if cid not in info:
            info[cid] = (d, m, None)

    RRF_K = 60
    rrf = {}
    for rank, cid in enumerate(vec_ids):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)
    for rank, (cid, _, _) in enumerate(bm_hits):
        rrf[cid] = rrf.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

    fused_ids = [cid for cid, _ in sorted(rrf.items(), key=lambda x: x[1], reverse=True)][:n_fetch]

    docs = [info[cid][0] for cid in fused_ids]
    metas = [info[cid][1] for cid in fused_ids]
    dists = [info[cid][2] for cid in fused_ids]

    missing = [i for i, d in enumerate(dists) if d is None]
    if missing:
        sims = _cosine_scores(q_emb, embedder.encode([docs[i] for i in missing]))
        for i, sim in zip(missing, sims):
            dists[i] = 1.0 - sim

    if not docs:
        return None

    if reranker is not None:
        pairs = [(q_text, d) for d in docs]
        scores = reranker.predict(pairs).tolist()
        order = sorted(range(len(docs)), key=lambda i: scores[i], reverse=True)
    else:
        scores = None
        order = sorted(range(len(docs)), key=lambda i: dists[i])

    if req.k:
        order = order[: req.k]
    docs = [docs[i] for i in order]
    metas = [metas[i] for i in order]
    dists = [dists[i] for i in order]
    rerank_scores = [scores[i] for i in order] if scores else [None] * len(docs)

    f_docs, f_metas, f_dists, f_rr = [], [], [], []
    budget = None if req.k else int(OLLAMA_NUM_CTX * 0.45)
    used = 0
    for d, m, dist, rr in zip(docs, metas, dists, rerank_scores):
        if 1.0 - float(dist) < req.min_score:
            continue
        if budget is not None:
            w = len(d.split())
            if f_docs and used + w > budget:
                break
            used += w
        f_docs.append(d)
        f_metas.append(m)
        f_dists.append(dist)
        f_rr.append(rr)

    if not f_docs:
        return []
    return f_docs, f_metas, f_dists, f_rr

def build_sources(docs, metas, dists, rrs) -> List[dict]:
    out = []
    for i, (d, m, dist, rr) in enumerate(zip(docs, metas, dists, rrs), 1):
        out.append({
            "id": i,
            "source": m.get("source", "unknown"),
            "chunk": m.get("chunk"),
            "score": 1.0 - float(dist),
            "rerank_score": rr,
            "text": d,
        })
    return out


def ingest_single_file(path: str) -> int:
    raw = load_any(path)
    chunks = split_text(raw, CHUNK_SIZE, CHUNK_OVERLAP)

    existing = collection.get(where={"source": path}, include=[])
    if existing["ids"]:
        collection.delete(ids=existing["ids"])

    if not chunks:
        return 0

    base_id = file_id(path)
    ids = [f"{base_id}-{idx:05d}" for idx in range(len(chunks))]
    metadatas = [{"source": path, "chunk": idx} for idx in range(len(chunks))]
    embeddings = embedder.encode(chunks, batch_size=64)

    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)
    return len(chunks)

_ingest_queue = queue.Queue()
_ingest_lock = threading.Lock()
_ingest_state = {
    "active": False,
    "total": 0,
    "current": None,
    "pending": [],
    "results": [],
}

def _ingest_worker():
    while True:
        path = _ingest_queue.get()
        with _ingest_lock:
            _ingest_state["current"] = path
            if path in _ingest_state["pending"]:
                _ingest_state["pending"].remove(path)
        try:
            n = ingest_single_file(path)
            if n > 0:
                result = {"source": path, "ok": True, "chunks": n}
            else:
                result = {"source": path, "ok": False, "error": "no extractable text"}
        except Exception as e:
            result = {"source": path, "ok": False, "error": str(e)}
        rebuild_bm25()
        with _ingest_lock:
            _ingest_state["results"].append(result)
            _ingest_state["current"] = None
            if _ingest_queue.empty() and not _ingest_state["pending"]:
                _ingest_state["active"] = False
        _ingest_queue.task_done()

threading.Thread(target=_ingest_worker, daemon=True, name="ingest-worker").start()

def enqueue_ingest(paths: List[str]):
    if not paths:
        return
    with _ingest_lock:
        if not _ingest_state["active"]:
            _ingest_state["results"] = []
            _ingest_state["total"] = 0
            _ingest_state["active"] = True
        _ingest_state["total"] += len(paths)
        _ingest_state["pending"].extend(paths)
    for p in paths:
        _ingest_queue.put(p)

def _needs_ingest(fpath: str) -> bool:
    existing = collection.get(where={"source": fpath}, include=[])
    if existing["ids"]:
        prefix = existing["ids"][0][:20]
        if prefix == file_id(fpath) or prefix == legacy_file_id(fpath):
            return False
    return True

WATCH_DOCS_INTERVAL = int(os.getenv("WATCH_DOCS_INTERVAL", "20"))
_watch_handled = {}

def _watch_docs():
    prev = {}
    handled = _watch_handled
    while True:
        try:
            current = {}
            for fpath in yield_files(DOCS_DIR):
                try:
                    st = os.stat(fpath)
                except OSError:
                    continue
                current[fpath] = (st.st_mtime, st.st_size)

            with _ingest_lock:
                inflight = set(_ingest_state["pending"])
                if _ingest_state["current"]:
                    inflight.add(_ingest_state["current"])

            to_add = []
            for p, sig in current.items():
                if prev.get(p) != sig:
                    continue
                if handled.get(p) == sig or p in inflight:
                    continue
                if _needs_ingest(p):
                    to_add.append(p)
                handled[p] = sig
            if to_add:
                print(f"[watch] auto-indexing {len(to_add)} new/changed file(s)")
                enqueue_ingest(to_add)
            prev = current
        except Exception as e:
            print(f"[watch] error: {e}")
        time.sleep(WATCH_DOCS_INTERVAL)

if WATCH_DOCS_INTERVAL > 0:
    threading.Thread(target=_watch_docs, daemon=True, name="docs-watcher").start()


@app.get("/")
def ui():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.get("/health")
def health():
    res = collection.get(include=["metadatas"])
    sources = {m.get("source", "unknown") for m in res["metadatas"]}
    model = OLLAMA_MODEL if LLM_BACKEND == "ollama" else OPENAI_MODEL
    return {
        "ok": True,
        "collection": COLLECTION_NAME,
        "backend": LLM_BACKEND,
        "model": model,
        "documents": len(sources),
        "chunks": len(res["metadatas"]),
        "llm": llm_status(),
        "indexing": _ingest_state["active"],
    }

@app.post("/upload")
def upload(files: List[UploadFile] = File(...)):
    os.makedirs(DOCS_DIR, exist_ok=True)
    queued = []
    rejected = []
    for uf in files:
        name = os.path.basename(uf.filename or "")
        name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip()
        if not name:
            rejected.append({"filename": uf.filename, "error": "invalid filename"})
            continue

        ext = os.path.splitext(name)[1].lower()
        if ext not in ALLOWED_EXTS:
            rejected.append({"filename": name, "error": f"unsupported file type '{ext}'"})
            continue

        dest = os.path.join(DOCS_DIR, name)
        try:
            with open(dest, "wb") as f:
                f.write(uf.file.read())
        except Exception as e:
            rejected.append({"filename": name, "error": str(e)})
            continue
        queued.append(dest)

    enqueue_ingest(queued)
    return {"queued": [{"filename": os.path.basename(p), "source": p} for p in queued],
            "rejected": rejected}

@app.post("/rescan")
def rescan():
    os.makedirs(DOCS_DIR, exist_ok=True)
    to_ingest = []
    unchanged = 0
    for fpath in yield_files(DOCS_DIR):
        existing = collection.get(where={"source": fpath}, include=[])
        if existing["ids"]:
            prefix = existing["ids"][0][:20]
            if prefix == file_id(fpath) or prefix == legacy_file_id(fpath):
                unchanged += 1
                continue
        to_ingest.append(fpath)
    enqueue_ingest(to_ingest)
    return {"queued": len(to_ingest), "unchanged": unchanged}

@app.post("/ingest-cancel")
def ingest_cancel():
    drained = []
    with _ingest_lock:
        while True:
            try:
                drained.append(_ingest_queue.get_nowait())
                _ingest_queue.task_done()
            except queue.Empty:
                break
        _ingest_state["pending"].clear()
        _ingest_state["total"] = len(_ingest_state["results"]) + (1 if _ingest_state["current"] else 0)
        if not _ingest_state["current"]:
            _ingest_state["active"] = False
    for p in drained:
        try:
            st = os.stat(p)
            _watch_handled[p] = (st.st_mtime, st.st_size)
        except OSError:
            pass
    return {"cancelled": len(drained)}

@app.get("/ingest-status")
def ingest_status():
    with _ingest_lock:
        return {
            "active": _ingest_state["active"],
            "total": _ingest_state["total"],
            "completed": len(_ingest_state["results"]),
            "current": _ingest_state["current"],
            "results": list(_ingest_state["results"]),
        }

@app.get("/documents")
def list_documents():
    res = collection.get(include=["metadatas"])
    counts = {}
    for m in res["metadatas"]:
        src = m.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    docs = [{"source": s, "chunks": c} for s, c in sorted(counts.items())]
    return {"documents": docs, "total_chunks": sum(counts.values())}

@app.get("/document-text")
def document_text(source: str, max_chars: int = 20000):
    res = collection.get(where={"source": source}, include=["documents", "metadatas"])
    if not res["ids"]:
        raise HTTPException(status_code=404, detail=f"No chunks found for source '{source}'")
    items = sorted(
        zip(res["documents"], res["metadatas"]),
        key=lambda x: x[1].get("chunk", 0),
    )
    text = "\n\n".join(d for d, _ in items)
    truncated = len(text) > max_chars
    return {
        "source": source,
        "chunks": len(items),
        "text": text[:max_chars],
        "truncated": truncated,
    }

@app.delete("/documents/all")
def delete_all_documents():
    res = collection.get(include=["metadatas"])
    ids = res["ids"]
    sources = {m.get("source", "") for m in res["metadatas"]}
    if ids:
        collection.delete(ids=ids)
    rebuild_bm25()

    files_deleted = 0
    real_docs = os.path.realpath(DOCS_DIR)
    for src in sources:
        real_src = os.path.realpath(src)
        if real_src.startswith(real_docs + os.sep) and os.path.isfile(real_src):
            os.remove(real_src)
            files_deleted += 1

    return {
        "documents_removed": len(sources),
        "chunks_removed": len(ids),
        "files_deleted": files_deleted,
    }

@app.delete("/documents")
def delete_document(source: str):
    existing = collection.get(where={"source": source}, include=[])
    if not existing["ids"]:
        raise HTTPException(status_code=404, detail=f"No chunks found for source '{source}'")
    collection.delete(ids=existing["ids"])
    rebuild_bm25()

    real_docs = os.path.realpath(DOCS_DIR)
    real_src = os.path.realpath(source)
    if real_src.startswith(real_docs + os.sep) and os.path.isfile(real_src):
        os.remove(real_src)

    return {"deleted": source, "chunks_removed": len(existing["ids"])}

@app.post("/ask", response_model=AskResp)
def ask(req: AskReq = Body(...)):
    retrieved = retrieve(req)
    if retrieved is None:
        return AskResp(answer=NO_CONTEXT_MSG, sources=[])
    if retrieved == []:
        return AskResp(answer=WEAK_MATCH_MSG, sources=[])

    docs, metas, dists, rrs = retrieved
    prompt = build_prompt(req.question, docs, style=req.answer_style)
    messages = build_messages(prompt, req.history)
    try:
        answer = llm_answer(messages, max_tokens=req.max_tokens, temperature=req.temperature)
    except requests.RequestException as e:
        raise HTTPException(
            status_code=502,
            detail=f"Could not reach the {LLM_BACKEND} LLM backend: {e}",
        )

    return AskResp(answer=answer, sources=build_sources(docs, metas, dists, rrs))

@app.post("/summarize")
def summarize(req: SummarizeReq = Body(...)):
    res = collection.get(where={"source": req.source}, include=["documents", "metadatas"])
    if not res["ids"]:
        raise HTTPException(status_code=404, detail=f"No chunks found for source '{req.source}'")
    items = sorted(zip(res["documents"], res["metadatas"]), key=lambda x: x[1].get("chunk", 0))
    full_text = "\n\n".join(d for d, _ in items)
    name = os.path.basename(req.source)

    def event(obj):
        return json.dumps(obj) + "\n"

    def gen():
        try:
            text = full_text
            words = text.split()
            if len(words) > 24000:
                partials = []
                for i in range(0, len(words), 8000):
                    seg = " ".join(words[i : i + 8000])
                    msgs = [
                        {"role": "system", "content": SUMMARY_SYSTEM},
                        {"role": "user", "content": f'Summarize the key points of this part of the document "{name}":\n\n{seg}'},
                    ]
                    partials.append(llm_answer(msgs, 600, req.temperature))
                text = "\n\n".join(partials)

            msgs = [
                {"role": "system", "content": SUMMARY_SYSTEM},
                {"role": "user", "content": (
                    f'Write a clear, well-structured summary of the document "{name}". '
                    "Use short sections or bullet points.\n\n"
                    f"Document content:\n\n{text}"
                )},
            ]
            if LLM_BACKEND == "ollama":
                for tok in stream_ollama(msgs, req.max_tokens, req.temperature):
                    yield event({"type": "token", "text": tok})
            else:
                yield event({"type": "token", "text": call_openai_like(msgs, req.max_tokens, req.temperature)})
            yield event({"type": "done"})
        except requests.RequestException as e:
            yield event({
                "type": "error",
                "message": f"Could not reach the {LLM_BACKEND} LLM backend: {e}",
            })

    return StreamingResponse(gen(), media_type="application/x-ndjson")

@app.post("/ask-stream")
def ask_stream(req: AskReq = Body(...)):
    def event(obj):
        return json.dumps(obj) + "\n"

    def gen():
        try:
            retrieved = retrieve(req)
        except Exception as e:
            yield event({"type": "error", "message": str(e)})
            return

        if retrieved is None or retrieved == []:
            msg = NO_CONTEXT_MSG if retrieved is None else WEAK_MATCH_MSG
            yield event({"type": "sources", "sources": []})
            yield event({"type": "token", "text": msg})
            yield event({"type": "done"})
            return

        docs, metas, dists, rrs = retrieved
        yield event({"type": "sources", "sources": build_sources(docs, metas, dists, rrs)})

        prompt = build_prompt(req.question, docs, style=req.answer_style)
        messages = build_messages(prompt, req.history)
        try:
            if LLM_BACKEND == "ollama":
                for tok in stream_ollama(messages, req.max_tokens, req.temperature):
                    yield event({"type": "token", "text": tok})
            else:
                answer = call_openai_like(messages, req.max_tokens, req.temperature)
                yield event({"type": "token", "text": answer})
            yield event({"type": "done"})
        except requests.RequestException as e:
            yield event({
                "type": "error",
                "message": f"Could not reach the {LLM_BACKEND} LLM backend: {e}",
            })

    return StreamingResponse(gen(), media_type="application/x-ndjson")
