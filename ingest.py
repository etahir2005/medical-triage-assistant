import os

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec


load_dotenv()

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
INDEX_NAME = "medical-triage"


def load_documents(data_folder="data"):
    documents = []
    for file_name in os.listdir(data_folder):
        if file_name.lower().endswith(".pdf"):
            file_path = os.path.join(data_folder, file_name)
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
    return documents


def split_documents(documents):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return text_splitter.split_documents(documents)


def store_in_pinecone(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5"
    )

    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = [i.name for i in pc.list_indexes()]
    if INDEX_NAME not in existing_indexes:
        pc.create_index(
            name=INDEX_NAME,
            dimension=768,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )

    vector_store = PineconeVectorStore.from_documents(
        documents=chunks,
        embedding=embeddings,
        index_name=INDEX_NAME
    )

    return vector_store


def main():
    print("Loading documents...")
    documents = load_documents()
    print(f"Loaded {len(documents)} pages")

    print("Splitting into chunks...")
    chunks = split_documents(documents)
    print(f"Created {len(chunks)} chunks")

    print("Storing in Pinecone...")
    store_in_pinecone(chunks)
    print("Done! All chunks stored in Pinecone.")


if __name__ == "__main__":
    main()