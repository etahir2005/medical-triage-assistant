"""Vector store retriever setup for the RAG pipeline."""

import logging

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

from src.config import EMBEDDING_MODEL_NAME, PINECONE_INDEX_NAME, RETRIEVAL_TOP_K

logger = logging.getLogger(__name__)


@st.cache_resource(show_spinner="Loading knowledge base...")
def get_retriever():
    """
    Load the embedding model and connect to the existing Pinecone index.

    Cached globally via @st.cache_resource so the embedding model loads
    once per server process, not once per user session.

    Returns:
        A LangChain retriever returning the top-k most similar chunks.

    Raises:
        RuntimeError: If the Pinecone index cannot be reached.
    """
    try:
        embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
        vector_store = PineconeVectorStore.from_existing_index(
            index_name=PINECONE_INDEX_NAME,
            embedding=embeddings,
        )
        return vector_store.as_retriever(search_kwargs={"k": RETRIEVAL_TOP_K})
    except Exception as exc:
        logger.exception("Failed to initialize retriever")
        raise RuntimeError(
            "Could not connect to the knowledge base. Please try again later."
        ) from exc
