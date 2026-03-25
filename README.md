# Customized RAG System

This project is designed to be modular and extendable. Which makes it easy to build more advanced workflows on top of the existing foundation.

Because the ingestion and serving pipelines are cleanly separated, you can swap embedding models, add alternative text splitters, integrate rerankers or modify prompt construction without disrupting the overall architecture.

This separation of concerns also allows the RAG server to accept new document ingestions at any time, enabling dynamic, evolving knowledge bases that stay up to date as you add or modify files in the `./docs` directory.

The system is also optimized for local, offline and privacy-preserving operation. All data remains on the user’s machine, eliminating reliance on external APIs and cloud services.

When paired with a locally hosted LLM such as those served by Ollama, the result is a fully self-contained personal knowledge assistant capable of answering questions, summarizing content and surfacing insights from your private documents with zero third-party exposure. This makes the project well-suited for researchers, engineers or anyone who needs secure access to their own information through natural language queries.

## Set Up Instructions

Below are the required software programs and instructions for installing and using this application.

### Programs Needed

- [Git](https://git-scm.com/downloads)

- [Python](https://www.python.org/downloads/)

### Steps For Use

1. Install the above programs

2. Open a terminal

3. Clone this repository using `git` by running the following command: `git clone git@github.com:devbret/retrieval-augmented-generation.git`

4. Navigate to the repo's directory by running: `cd retrieval-augmented-generation`

5. Create a virtual environment with this command: `python3 -m venv venv`

6. Activate your virtual environment using: `source venv/bin/activate`

7. Install the needed dependencies for running the script: `pip install -r requirements.txt`

8. Add your files to the `docs` directory within this repo

9. Convert the `.env.template` file into a `.env` file with your correct values

10. Launch your local LLM using Ollama: `ollama run mistral`

11. Ingest your documents with the following command: `python ingest.py`

12. Start the RAG server: `uvicorn rag_server:app --reload`

13. Confirm the RAG API is working correctly: `GET http://127.0.0.1:8000`

## Other Considerations

This project repo is intended to demonstrate an ability to do the following:

- Query private documents using natural language

- Regularly ingest and index new files added to the `docs` folder

If you have any questions or would like to collaborate, please reach out either on GitHub or via [my website](https://bretbernhoft.com/).
