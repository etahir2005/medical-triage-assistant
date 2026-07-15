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
- FastAPI — HTTP API
- Python

## How to Run
1. Clone the repo
2. Create a virtual environment and install dependencies
   pip install -r requirements.txt
3. Copy `.env.example` to `.env` and add your API keys
   GEMINI_API_KEY=your_key
   PINECONE_API_KEY=your_key
   API_ACCESS_KEY=your_key
   ALLOWED_ORIGINS=http://localhost:8501
   STREAMLIT_ACCESS_KEY=your_key
4. Add your PDF documents to the data/ folder
5. Run ingestion
   python scripts/ingest.py
6. Launch the app
   streamlit run app.py

## Run with Docker

Build and run each service individually:
docker build -f docker/api.Dockerfile -t triage-api .
docker run --env-file .env -p 8000:8000 triage-api
docker build -f docker/streamlit.Dockerfile -t triage-ui .
docker run --env-file .env -p 8501:8501 triage-ui

Or run both together with Docker Compose:
docker-compose up --build
This starts the API on `http://localhost:8000` and the Streamlit UI on
`http://localhost:8501`, both reading configuration from `.env`.

A few deliberate choices behind these images, worth knowing before
"fixing" them back to defaults:
- **CPU-only PyTorch** — installed via `--extra-index-url
  https://download.pytorch.org/whl/cpu`, since the default GPU build
  would add several GB to the image for a dependency this app never
  uses.
- **Offline-baked embedding model** — `BAAI/bge-base-en-v1.5` is
  downloaded once at build time, not at container startup, so the
  containers work with no network access and start faster.
- **venv over `--user` installs** — dependencies are installed into
  a virtualenv at `/opt/venv` and copied into the final stage rather
  than using `pip install --user`, since the non-root `appuser`'s
  `$HOME` isn't guaranteed to resolve consistently across every
  runtime context.

## Expected Inputs
- Free-text symptom descriptions typed into the chat box, in
  English, Roman Urdu, or Urdu script (e.g. "high fever",
  "mujhe bukhaar hai", or "مجھے بخار ہے")
- Maximum message length: 1000 characters
- Source PDFs for ingestion: WHO clinical guidelines and Pakistan
  Ministry of Health / NIH documents, placed in `data/`

## Expected Outputs
- A triage-guidance answer in the same language as the question,
  grounded only in the ingested source documents
- A faithfulness badge (✅ faithful / ⚠️ partial / ❌ low confidence)
  indicating how well the answer is supported by retrieved context
- Expandable source citations showing which document chunks were
  used to generate the answer
- For high-risk symptoms (e.g. chest pain, severe bleeding,
  difficulty breathing), an immediate emergency alert directing the
  user to call Rescue 1122, bypassing the RAG pipeline entirely

## Model Information
- LLM: Google Gemini 3.1 Flash Lite (generation, translation, and
  faithfulness grading)
- Embedding model: BAAI/bge-base-en-v1.5, run locally via
  Hugging Face, 768-dimensional vectors
- Vector store: Pinecone (serverless, AWS us-east-1, cosine
  similarity)

## Project Structure
medical-triage-assistant/
├── app.py                  # Streamlit entry point (UI only)
├── api.py                  # FastAPI entry point (HTTP API)
├── src/
│   ├── config.py           # Centralized constants and settings
│   ├── utils.py             # Shared helpers (language detection, etc.)
│   ├── schemas.py           # Pydantic request/response models
│   ├── chains.py           # Translation + answer generation (Gemini)
│   ├── grader.py            # Faithfulness grading
│   ├── retriever.py         # Pinecone retriever setup
│   ├── emergency.py         # Keyword-based emergency detection
│   └── pipeline.py          # Orchestrates the full request pipeline
├── scripts/
│   └── ingest.py            # One-time document ingestion into Pinecone
├── tests/
│   ├── test_emergency.py
│   ├── test_utils.py
│   └── test_api.py
├── data/                    # Source PDFs for ingestion
├── .env.example
├── requirements.txt
└── README.md

## HTTP API

In addition to the Streamlit UI, the project exposes a REST API.

Run it:
uvicorn api:app --reload --port 8000

Interactive docs: http://localhost:8000/docs

### POST /chat
Request:
```json
{
  "message": "high fever",
  "history": []
}
```

Response:
```json
{
  "is_emergency": false,
  "answer": "...",
  "grade": "faithful",
  "sources": [{"content": "...", "source": "who_guideline.pdf"}]
}
```

`API_ACCESS_KEY` is required — generate one, set it in `.env`, and
include it as an `X-API-Key` header on every `/chat` request.

### GET /health
Returns `{"status": "ok"}` — used for uptime monitoring.

## Streamlit UI Access
The UI requires `STREAMLIT_ACCESS_KEY` to be set in `.env`. Visitors
must enter this value before the chat interface becomes usable.

## Future Improvements
- Add a minimum similarity-score threshold on retrieval to filter
  out low-relevance chunks before generation
- Expand automated test coverage to the retrieval/generation/
  grading pipeline (currently covered by manual testing plus unit
  tests on the deterministic emergency-detection, language-
  detection, and API layer logic)
- Move source PDFs out of git history (e.g. Git LFS or a documented
  external download step) to keep the repository lightweight
- Add authentication/rate-limiting middleware to the API beyond the
  optional static API key, if exposed publicly