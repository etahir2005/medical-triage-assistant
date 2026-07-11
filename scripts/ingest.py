"""One-time ingestion pipeline: load PDFs, chunk, embed, and upload to Pinecone."""

import logging
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_NAME,
    INGESTION_BATCH_SIZE,
    PINECONE_API_KEY,
    PINECONE_INDEX_NAME,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


def load_documents(data_folder: str = "data") -> list:
    """
    Load all PDFs in a folder into Document objects.

    Args:
        data_folder: Path to the folder containing source PDFs.

    Returns:
        Loaded Document objects. Unreadable files are skipped and logged
        rather than aborting the whole run.
    """
    documents = []
    for file_name in os.listdir(data_folder):
        if not file_name.lower().endswith(".pdf"):
            continue
        file_path = os.path.join(data_folder, file_name)
        try:
            documents.extend(PyPDFLoader(file_path).load())
        except Exception:
            logger.exception("Skipping unreadable PDF: %s", file_name)
    return documents


def split_documents(documents: list) -> list:
    """
    Split documents into overlapping chunks for embedding.

    Args:
        documents: Loaded Document objects.

    Returns:
        Document chunks ready for embedding.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def _upsert_batch(vector_store, batch):
    """Upload a single batch of chunks, retrying on transient failures."""
    vector_store.add_documents(batch)


def store_in_pinecone(chunks: list):
    """
    Create the Pinecone index if needed and upload chunks in retry-safe batches.

    Args:
        chunks: Document chunks to embed and store.

    Returns:
        The connected PineconeVectorStore.
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [i.name for i in pc.list_indexes()]
    if PINECONE_INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=PINECONE_INDEX_NAME,
            dimension=EMBEDDING_DIMENSION,
            metric="cosine",
            spec=ServerlessSpec(cloud="aws", region="us-east-1"),
        )

    vector_store = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME, embedding=embeddings
    )

    total_batches = -(-len(chunks) // INGESTION_BATCH_SIZE)
    for i in range(0, len(chunks), INGESTION_BATCH_SIZE):
        batch = chunks[i : i + INGESTION_BATCH_SIZE]
        _upsert_batch(vector_store, batch)
        logger.info(
            "Uploaded batch %d/%d", i // INGESTION_BATCH_SIZE + 1, total_batches
        )

    return vector_store


def main():
    """Run the full ingestion pipeline end to end."""
    logger.info("Loading documents...")
    documents = load_documents()
    logger.info("Loaded %d pages", len(documents))

    logger.info("Splitting into chunks...")
    chunks = split_documents(documents)
    logger.info("Created %d chunks", len(chunks))

    logger.info("Storing in Pinecone...")
    store_in_pinecone(chunks)
    logger.info("Done! All chunks stored in Pinecone.")


if __name__ == "__main__":
    main()
