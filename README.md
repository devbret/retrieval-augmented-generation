This project is a lightweight Retrieval-Augmented Generation (RAG) system designed to let you query your own documents using a local language model. The pipeline begins with `ingest.py`, which scans a `./docs` directory for files, splits them into overlapping text chunks and generates embeddings using a SentenceTransformer model.

The second half of the project, `rag-server.py`, exposes a FastAPI service that accepts user questions and performs retrieval along with LLM answering. When a query comes in, the server encodes it into an embedding, searches the ChromaDB collection for the most relevant document chunks and then builds a prompt combining those results with the user’s question.

Depending on configuration, the system calls either an OpenAI-like backend or a locally running Ollama model to generate an answer grounded in the retrieved context. The result includes both the final answer and citations pointing back to document sources, ensuring transparency and traceability.

Together, the ingestion and server scripts form a self-contained local RAG system for private, document-aware AI assistance.
