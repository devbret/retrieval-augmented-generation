# Personal Knowledge RAG System

![Screenshot from the RAG GUI.](https://hosting.photobucket.com/bbcfb0d4-be20-44a0-94dc-65bff8947cf2/27c02ed7-1b50-4ab6-8a33-2c6b5c03bb77.png)

Self-hosted Retrieval-Augmented Generation system which turns documents into a private knowledge base accessible through a web UI.

## Application Overview

Documents enter the system through the `./docs` directory, either uploaded from the web UI or copied manually. An indexer extracts their text, splits it into overlapping chunks, generates embeddings and stores everything in a Chroma vector database. Large batches index without blocking the app, with live progress and a cancel option.

When you ask a question, the `FastAPI` server searches your library two ways: (1) semantic vector search catches meaning and (2) BM25 keyword index catches exact phrases, acronyms and document numbers. The two results are combined, re-ordered and are handed to a locally hosted language model. Answers stream back token by token with numbered citations.

The built-in web UI is the primary face of the system. It handles drag-and-drop uploads, follow-up questions, clickable citations to reveal the exact passage behind every claim and chat exports to CSV, JSON or HTML. Because the server binds to your network, the whole system can run on a dedicated AI PC while you use it from a browser on any other machine.

The pieces stay modular and configurable. Embedding model, reranker, chunk sizes, context window and the folder watcher are all set in the `.env`. Also answers can be restricted to specific documents from the UI and files added to `./docs` outside the UI are picked up by the automatic folder watcher.

## Basic Setup Instructions

Below are the required software programs and instructions for installing and using this application on a Linux machine.

### Programs Needed

- [Git](https://git-scm.com/downloads)

- [Python](https://www.python.org/downloads/)

- [Tesseract](https://github.com/tesseract-ocr/tesseract)

- [Ollama](https://ollama.com/download)

### Steps For Use

1. Install the above programs

2. Open a terminal

3. Clone this repository: `git clone git@github.com:devbret/retrieval-augmented-generation.git`

4. Navigate to the repo's directory: `cd retrieval-augmented-generation`

5. Create a virtual environment: `python3 -m venv venv`

6. Activate your virtual environment: `source venv/bin/activate`

7. Install the needed dependencies: `pip install -r requirements.txt`

8. Convert the `.env.template` file into a `.env` file: `cp .env.template .env`

9. Add values to the `.env` file: `nano .env`

10. Start the RAG server: `uvicorn rag_server:app --host 0.0.0.0 --port 47821`

11. Open the web UI in a browser: `http://127.0.0.1:47821`

12. Add your files via the web UI and begin chatting

## Other Considerations

This project repo is intended to demonstrate an ability to do the following:

- Turn a folder of personal documents into a private searchable knowledge base

- Answer natural language questions with streamed responses which cite their sources

- Combine semantic vector search, BM25 keyword matching and reranking

- Serve a full web UI to any browser on the local network without sending data beyond it

If you have any questions or would like to collaborate, please reach out either on GitHub or via [my website](https://bretbernhoft.com/).
