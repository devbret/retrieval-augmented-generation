# Customized RAG System

This project is designed to be modular and extendable. Which makes it easy to build more advanced workflows on top of the existing foundation.

Because the ingestion and serving pipelines are cleanly separated, you can swap embedding models, add alternative text splitters, integrate rerankers or modify prompt construction without disrupting the overall architecture.

This separation of concerns also allows the RAG server to accept new document ingestions at any time, enabling dynamic, evolving knowledge bases that stay up to date as you add or modify files in the `./docs` directory.

The system is also optimized for local, offline and privacy-preserving operation. All data remains on the user’s machine, eliminating reliance on external APIs and cloud services.

When paired with a locally hosted LLM such as those served by Ollama, the result is a fully self-contained personal knowledge assistant capable of answering questions, summarizing content and surfacing insights from your private documents with zero third-party exposure. This makes the project well-suited for researchers, engineers or anyone who needs secure access to their own information through natural language queries.
