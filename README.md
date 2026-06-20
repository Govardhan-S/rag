# Simple RAG — Beginner Friendly

This small project demonstrates a beginner-friendly Retrieval-Augmented Generation
(RAG) pipeline using Python, Streamlit, ChromaDB, Ollama, and simple text processing.

## What this project shows
- Upload a TXT file and extract text.
- Split the text into chunks (chunk size 500, overlap 50).
- Create embeddings for chunks using Ollama's `nomic-embed-text` model.
- Store vectors and chunks in a local ChromaDB database.
- Retrieve the top-3 relevant chunks for a user question and show them.
- Generate a final answer with Ollama's `llama3.2:1b` using only the retrieved context.

## Files
- `app.py` — Streamlit user interface (upload, process, ask questions).
- `rag.py` — Helper functions: text extraction, chunking, embedding calls,
	ChromaDB storage and retrieval, and calling Ollama for generation.
- `requirements.txt` — Python dependencies.
- `chroma_db/` — Folder where ChromaDB persists the vector database (created at runtime).

## Quick setup (Windows PowerShell)

1) Create and activate a virtual environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -U pip
```

2) Install Python dependencies

```powershell
pip install -r requirements.txt
```

3) Run Ollama and pull required models

Start your Ollama server following Ollama's documentation. Then pull the models used by
this project:

```powershell
ollama pull nomic-embed-text
ollama pull llama3.2:1b
```

4) Set `OLLAMA_URL` if your Ollama server is not at the default `http://localhost:11434`

```powershell
setx OLLAMA_URL "http://your-ollama-host:11434"
```

5) Run the Streamlit app

```powershell
streamlit run app.py
```

Open the URL Streamlit shows (usually http://localhost:8501) in your browser.

## How to use the app
1. Upload a TXT file in the "Upload a TXT file" area.
2. Click "Process Document" — this extracts text, creates chunks, embeds them, and stores them in ChromaDB.
3. Enter a question in the question box and click "Ask Question".
4. The app will display the top-3 retrieved chunks and the generated answer.

## Short explanation of RAG concepts (for students)
- Chunking: splitting long documents into smaller, overlapping pieces so models can
	process them within token or length limits while keeping context.
- Embeddings: numeric vector representations of text that capture semantic meaning;
	used to find similar pieces of text.
- ChromaDB: a lightweight vector store that persists embeddings and documents and
	supports efficient similarity search (retrieval).
- Retrieval: finding the most relevant chunks for a user query using vector similarity.
- Generation: an LLM composes the final answer using only the retrieved context and the
	user question.

## Notes and troubleshooting
- Ensure Ollama is running and accessible at the `OLLAMA_URL` used by the app.
- The `chroma_db/` directory is created automatically and stores the DB; you can inspect
	it or remove it to start fresh.
- This project is intentionally minimal and uses simple functions to be easy to read.

If you want, I can add a `README` section that includes screenshots or expand the
explanations further.
# rag