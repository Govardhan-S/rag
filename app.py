import streamlit as st
from rag import (
    extract_text,
    chunk_text,
    get_embeddings,
    store_in_chroma,
    retrieve_top_k,
    generate_answer,
)

# Simple Streamlit app for a beginner-friendly RAG demo

st.set_page_config(page_title="Simple RAG Demo", layout="wide")

st.title("Simple RAG — Beginner Friendly")

st.markdown(
    """
    This example shows a small Retrieval-Augmented Generation (RAG) pipeline:
    upload a TXT file, chunk it, embed chunks, store them in ChromaDB, then
    retrieve and generate answers with Ollama.
    """
)

uploaded_file = st.file_uploader(
    "Upload a TXT file",
    type=["txt"]
)
if "chunks" not in st.session_state:
    st.session_state["chunks"] = []

if "processed" not in st.session_state:
    st.session_state["processed"] = False

if uploaded_file:
    st.write(f"Uploaded: {uploaded_file.name}")

    if st.button("Process Document"):
        try:
            # Step 1: read the uploaded file
            text = extract_text(uploaded_file)

            # Step 2: chunk text
            chunks = chunk_text(text, chunk_size=500, chunk_overlap=50)
            st.session_state["chunks"] = chunks
            st.write(f"Created {len(chunks)} chunks")

            # Step 3 & 4: embeddings and store in ChromaDB
            st.info("Generating embeddings and storing in ChromaDB — this may take a moment.")
            embeddings = get_embeddings(chunks)
            store_in_chroma(chunks, embeddings)

            st.session_state["processed"] = True
            st.success("Document processed and stored in ChromaDB.")
        except ValueError as e:
            st.error(str(e))
        except Exception as e:
            st.error("An unexpected error occurred while processing the document.")
            st.exception(e)

st.markdown("---")

st.header("Ask a Question")
question = st.text_input("Enter your question")

if st.button("Ask Question"):
    if not st.session_state.get("processed"):
        st.warning("Please upload and process a document first.")
    elif not question:
        st.warning("Please enter a question.")
    else:
        # Step 6: retrieval
        results = retrieve_top_k(question, k=3)

        st.subheader("Retrieved Chunks (top 3)")
        for i, doc in enumerate(results):
            st.markdown(f"**Chunk {i+1}**")
            st.write(doc.get("text", ""))

        # Step 7: generation
        context = "\n\n".join([r.get("text", "") for r in results])
        answer = generate_answer(context, question)

        st.subheader("Final Answer")
        st.write(answer)
