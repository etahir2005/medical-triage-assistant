# Medical Triage Assistant

A bilingual RAG-based medical triage chatbot that helps users in 
Pakistan assess symptoms and decide what to do next. Supports 
English, Roman Urdu and Urdu script.

## What It Does
- Answers symptom questions grounded in WHO and Pakistan Ministry 
  of Health guidelines
- Responds in the same language the user writes in
- Grades every answer for faithfulness to the source documents
- Detects emergencies instantly and directs users to call 1122
- Shows source citations for every answer

## Tech Stack
- LangChain — RAG pipeline and prompt management
- Pinecone — vector database for document storage and retrieval
- BAAI/bge-base-en-v1.5 — local embedding model
- Google Gemini 3.1 Flash Lite — LLM for generation and grading
- Streamlit — chat interface
- Python

## How to Run
1. Clone the repo
2. Create a virtual environment and install dependencies
   pip install -r requirements.txt
3. Add your API keys to a .env file
   GEMINI_API_KEY=your_key
   PINECONE_API_KEY=your_key
4. Add your PDF documents to the data/ folder
5. Run ingestion
   python ingest.py
6. Launch the app
   streamlit run app.py
