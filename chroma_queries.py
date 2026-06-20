"""Basic ChromaDB query examples for the current project.

This file shows how to:
- connect to a persistent ChromaDB client
- create or load a collection
- add documents and embeddings
- query by embedding
- read stored documents and metadata
- list collections and clear data

It is intentionally simple for beginners.
"""

import os
import requests
import chromadb
from chromadb.config import Settings

# Configuration for Ollama and ChromaDB
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
EMBED_MODEL = "nomic-embed-text"
CHROMA_DIR = os.path.join(os.getcwd(), "chroma_db")


def create_chroma_client():
    """Create a persistent ChromaDB client using the new API."""
    return chromadb.PersistentClient(
        path=CHROMA_DIR,
        settings=Settings(
            chroma_api_impl="chromadb.api.rust.RustBindingsAPI",
            persist_directory=CHROMA_DIR,
        ),
    )


def create_collection(client, collection_name="example_collection"):
    """Create or open a Chroma collection."""
    return client.get_or_create_collection(name=collection_name)


def ollama_embed(texts):
    """Generate embeddings from Ollama for a list of texts."""
    url = f"{OLLAMA_URL}/api/embed"
    payload = {"model": EMBED_MODEL, "input": texts}
    response = requests.post(url, json=payload)
    response.raise_for_status()
    data = response.json()
    if "embeddings" in data:
        return data["embeddings"]
    raise ValueError(f"Unexpected Ollama response: {data}")


def add_documents(collection, documents, metadatas=None):
    """Add documents and embeddings to a Chroma collection."""
    if metadatas is None:
        metadatas = [{} for _ in documents]

    embeddings = ollama_embed(documents)
    ids = [f"doc_{i}" for i in range(len(documents))]

    collection.add(
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids,
    )
    return ids


def query_collection(collection, query_text, k=3):
    """Query the collection using a text prompt and return top-k results."""
    query_embedding = ollama_embed([query_text])[0]
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def get_all_documents(collection):
    """Return all stored documents and metadata."""
    return collection.get(include=["documents", "metadatas"])


def get_documents_by_ids(collection, ids):
    """Return specific stored documents by id."""
    return collection.get(ids=ids, include=["documents", "metadatas"])


def list_collections(client):
    """List all collections in the client."""
    return [c.name for c in client.list_collections()]


def delete_collection(client, collection_name):
    """Delete a named collection and its data."""
    client.delete_collection(name=collection_name)


def clear_collection(collection):
    """Remove all documents from an existing collection."""
    collection.delete()


def main():
    client = create_chroma_client()
    print("Connected to ChromaDB at:", CHROMA_DIR)

    collection_name = "example_collection"
    collection = create_collection(client, collection_name)
    print("Loaded collection:", collection_name)

    sample_docs = [
        "The library is open from 8 AM to 8 PM.",
        "Students can borrow up to 4 books for 14 days.",
        "Attendance must be at least 75 percent.",
    ]
    sample_metadata = [
        {"section": "library"},
        {"section": "library"},
        {"section": "attendance"},
    ]

    print("Adding sample documents...")
    ids = add_documents(collection, sample_docs, sample_metadata)
    print("Added documents with ids:", ids)

    print("\nStored documents:")
    stored = get_all_documents(collection)
    print(stored)

    query_text = "What are the library hours?"
    print(f"\nQuerying for: {query_text}")
    results = query_collection(collection, query_text, k=3)
    print(results)

    print("\nDocuments by id:")
    print(get_documents_by_ids(collection, ids[:2]))

    print("\nCollections:")
    print(list_collections(client))

    # Uncomment the next line to delete all data in this collection.
    # clear_collection(collection)
    # delete_collection(client, collection_name)


if __name__ == "__main__":
    main()
