import os, hashlib
from dotenv import load_dotenv
load_dotenv()

import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from util_text import load_any, split_text, yield_files

CHROMA_DIR = os.getenv("CHROMA_DIR", "./chroma_db")
EMBED_MODEL = os.getenv("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "local_rag")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "800"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "120"))

DOCS_DIR = "./docs"

def file_id(path: str) -> str:
    h = hashlib.sha256()
    h.update(os.path.abspath(path).encode())
    h.update(str(os.path.getmtime(path)).encode())
    return h.hexdigest()[:20]

def ingest_path(docs_dir: str = DOCS_DIR) -> int:
    os.makedirs(CHROMA_DIR, exist_ok=True)

    client = chromadb.PersistentClient(
        path=CHROMA_DIR, settings=Settings(allow_reset=False)
    )

    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    embedder = SentenceTransformer(EMBED_MODEL)

    to_add_texts, to_add_ids, to_add_metadatas = [], [], []

    for fpath in yield_files(docs_dir):
        try:
            raw = load_any(fpath)
        except Exception as e:
            print(f"[skip] {fpath}: {e}")
            continue

        chunks = split_text(raw, CHUNK_SIZE, CHUNK_OVERLAP)
        if not chunks:
            continue

        base_id = file_id(fpath)
        for idx, ch in enumerate(chunks):
            doc_id = f"{base_id}-{idx:05d}"
            to_add_ids.append(doc_id)
            to_add_texts.append(ch)
            to_add_metadatas.append({"source": fpath, "chunk": idx})

    if not to_add_texts:
        print(f"No documents found to index in {docs_dir}.")
        return 0

    print(f"Embedding {len(to_add_texts)} chunks with {EMBED_MODEL} ...")
    embeddings = embedder.encode(to_add_texts, batch_size=64, show_progress_bar=True)

    print("Writing to Chroma…")
    collection.add(
        ids=to_add_ids,
        embeddings=embeddings,
        documents=to_add_texts,
        metadatas=to_add_metadatas,
    )

    print(f"Done. Collection: {COLLECTION_NAME} at {CHROMA_DIR}")
    return len(to_add_texts)

def main():
    count = ingest_path(DOCS_DIR)
    print(f"Ingested {count} chunks from {DOCS_DIR}.")

if __name__ == "__main__":
    main()
