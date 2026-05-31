# Personal Knowledge RAG System

A modular, local-first Retrieval-Augmented Generation application for building a private personal knowledge assistant.

## Application Overview

The application works in two separate stages. The ingestion pipeline scans the `./docs` directory, loads supported document types, splits their contents into manageable chunks, generates embeddings and stores those chunks in a persistent Chroma vector database.

Whereas the serving pipeline exposes a FastAPI-based RAG server to accept natural language questions, then embeds each query, retrieves the most relevant document chunks and uses those chunks as context for a locally hosted language model. The server can work with either an OpenAI-compatible endpoint or an `Ollama` model.

Because ingestion and serving are separated, the project is easy to extend. You can swap embedding models, adjust chunking behavior, filter answers by document source or modify prompt construction without disrupting the rest of the system. New or updated files can be added to the `./docs` directory and re-ingested.

## Basic Setup Instructions

Below are the required software programs and instructions for installing and using this application on a Linux machine.

### Programs Needed

- [Git](https://git-scm.com/downloads)

- [Python](https://www.python.org/downloads/)

### Steps For Use

1. Install the above programs

2. Open a terminal

3. Clone this repository: `git clone git@github.com:devbret/retrieval-augmented-generation.git`

4. Navigate to the repo's directory: `cd retrieval-augmented-generation`

5. Create a virtual environment: `python3 -m venv venv`

6. Activate your virtual environment: `source venv/bin/activate`

7. Install the needed dependencies: `pip install -r requirements.txt`

8. Add your files to the `docs` directory within this repo

9. Convert the `.env.template` file into a `.env` file: `cp .env.template .env`

10. Add values to the `.env` file: `nano .env`

11. Launch your local LLM using Ollama: `ollama run mistral`

12. Ingest your documents with the following command: `python3 ingest.py`

13. Start the RAG server: `uvicorn rag_server:app --reload`

14. Confirm the RAG API is working correctly: `GET http://127.0.0.1:8000`

## Other Considerations

This project repo is intended to demonstrate an ability to do the following:

- Turn a local folder of private documents into a searchable knowledge base

- Enable users to ask natural language questions and receive answers grounded in their own files

- Keep the ingestion, retrieval and language model serving layers modular so each part can be easily upgraded

- Operate a fully local RAG assistant paired with Ollama or another local LLM backend

If you have any questions or would like to collaborate, please reach out either on GitHub or via [my website](https://bretbernhoft.com/).
