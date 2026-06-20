"""
Simple RAG helper functions.

This file contains functions to:
- extract text from a PDF or TXT file
- chunk text into overlapping pieces
- get embeddings from Ollama's nomic-embed-text model
- store and query a local ChromaDB collection
- generate an answer using Ollama's llama3.2:1b model
"""

from typing import List
import chromadb
import requests
import os
from io import BytesIO
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError
from langchain_text_splitters import CharacterTextSplitter
from chromadb.config import Settings

# ----------------------
# Configuration
# ----------------------
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.2:1b"
CHROMA_DIR = os.path.join(os.getcwd(), "chroma_db")


# ----------------------
# File Processing
# ----------------------
# def extract_text_from_pdf(file) -> str:
#     """
#     Extract text from PDF or TXT files.

#     Supported formats:
#     - PDF (.pdf)
#     - Text (.txt)
#     """

#     filename = getattr(file, "name", "")

#     try:
#         # Uploaded file from Streamlit
#         if hasattr(file, "read"):
#             file.seek(0)
#             data = file.read()
#         else:
#             # Local file path
#             with open(file, "rb") as f:
#                 data = f.read()

#         if not data:
#             return ""

#         # ----------------------
#         # TXT files
#         # ----------------------
#         if filename.lower().endswith(".txt"):
#             try:
#                 return data.decode("utf-8")
#             except UnicodeDecodeError:
#                 return data.decode("latin-1", errors="replace")

#         # ----------------------
#         # PDF files
#         # ----------------------
#         if filename.lower().endswith(".pdf"):
#             try:
#                 reader = PdfReader(BytesIO(data), strict=False)

#                 pages = []
#                 for page in reader.pages:
#                     try:
#                         pages.append(page.extract_text() or "")
#                     except Exception:
#                         continue

#                 return "\n\n".join(pages)

#             except Exception as e:
#                 print(f"PDF parsing error: {e}")
#                 return ""

#         raise ValueError(
#             f"Unsupported file type: {filename}. Only .pdf and .txt files are supported."
#         )

#     except Exception as e:
#         print(f"File processing error: {e}")
#         return ""

def extract_text(file):
    filename = getattr(file, "name", "")

    if hasattr(file, "seek"):
        file.seek(0)

    if hasattr(file, "getvalue"):
        data = file.getvalue()
    else:
        data = file.read()

    if not data:
        raise ValueError("Uploaded file is empty.")

    if filename.lower().endswith(".txt"):
        return data.decode("utf-8", errors="replace")

    raise ValueError("Unsupported file type. Please upload a .txt file.")
# ----------------------
# Text Chunking
# ----------------------
def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50
) -> List[str]:
    """
    Split text into chunks.

    Chunking divides long text into smaller pieces so embeddings
    and LLMs can process them effectively.
    """

    splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    return splitter.split_text(text)


# ----------------------
# Embeddings
# ----------------------
def _ollama_request(method: str, paths: List[str], payload: dict):
    """Send a request to Ollama and try fallback API paths."""

    errors = []
    base_url = OLLAMA_URL.rstrip("/")

    for path in paths:
        url = f"{base_url}{path}"
        try:
            response = requests.request(method, url, json=payload, timeout=30)
        except requests.RequestException as exc:
            errors.append(f"{url}: {exc}")
            continue

        if response.status_code == 200:
            return response

        if response.status_code in (404, 405):
            errors.append(f"{url} returned {response.status_code}")
            continue

        try:
            response.raise_for_status()
        except requests.HTTPError as exc:
            raise ValueError(
                f"Ollama request to {url} failed: {response.status_code} {response.text}"
            ) from exc

    raise ValueError(
        "Unable to reach a compatible Ollama endpoint. "
        f"Tried: {', '.join(paths)}. Details: {' | '.join(errors)}"
    )


def ollama_embed(texts: List[str]) -> List[List[float]]:
    """
    Generate embeddings using Ollama's embedding API.
    """

    url = f"{OLLAMA_URL}/api/embed"

    payload = {
        "model": EMBED_MODEL,
        "input": texts
    }

    response = requests.post(url, json=payload)
    response.raise_for_status()

    data = response.json()

    if "embeddings" in data:
        return data["embeddings"]

    raise ValueError(f"Unexpected Ollama response: {data}")


def get_embeddings(chunks: List[str]) -> List[List[float]]:
    """
    Generate embeddings for all text chunks.
    """
    return ollama_embed(chunks)


# ----------------------
# ChromaDB
# ----------------------
def _get_chroma_client():
    """
    Create a persistent ChromaDB client using the new client API.
    """

    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(
            chroma_api_impl="chromadb.api.rust.RustBindingsAPI",
            persist_directory=CHROMA_DIR,
        ),
    )


def store_in_chroma(
    chunks: List[str],
    embeddings: List[List[float]],
    collection_name: str = "rag_collection"
):
    """
    Store chunks and embeddings in ChromaDB.
    """

    client = _get_chroma_client()

    collection = client.get_or_create_collection(
        name=collection_name
    )

    ids = [f"doc_{i}" for i in range(len(chunks))]
    metadatas = [{"chunk_index": i} for i in range(len(chunks))]

    collection.add(
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )


def retrieve_top_k(
    query: str,
    k: int = 3,
    collection_name: str = "rag_collection"
) -> List[dict]:
    """
    Retrieve the most relevant chunks from ChromaDB.
    """

    query_embedding = ollama_embed([query])[0]

    client = _get_chroma_client()

    collection = client.get_or_create_collection(
        name=collection_name
    )

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "distances"]
    )

    docs = []

    documents = results.get("documents", [])
    distances = results.get("distances", [])

    if documents:
        for i, doc in enumerate(documents[0]):
            docs.append({
                "text": doc,
                "score": distances[0][i] if distances else None
            })

    return docs


# ----------------------
# Answer Generation
# ----------------------
def generate_answer(context: str, question: str) -> str:
    """
    Generate an answer using retrieved context.
    """

    prompt = f"""
You are a helpful AI assistant.

Answer the question using only the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False
    }

    response = _ollama_request(
        "post",
        ["/api/generate", "/v1/generate", "/generate"],
        payload,
    )

    data = response.json()

    if "response" in data:
        return data["response"]

    return str(data)