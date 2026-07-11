import os

from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical-triage"

def get_retriever():
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    vector_store = PineconeVectorStore.from_existing_index(
        index_name=INDEX_NAME,
        embedding=embeddings
    )

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 5}
    )

    return retriever