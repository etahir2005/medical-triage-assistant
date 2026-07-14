"""FastAPI entry point for the Medical Triage Assistant HTTP API."""

import logging
import os

from fastapi import Depends, FastAPI, Header, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware

from src.pipeline import process_user_message
from src.retriever import get_retriever
from src.schemas import ChatRequest, ChatResponse, HealthResponse, SourceDocument

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

API_ACCESS_KEY = os.getenv("API_ACCESS_KEY")
if not API_ACCESS_KEY:
    raise EnvironmentError(
        "API_ACCESS_KEY is not set. This API refuses to start without "
        "it — running without authentication would let anyone consume "
        "your Gemini and Pinecone quota. Add it to your .env file."
    )

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")

app = FastAPI(
    title="Medical Triage Assistant API",
    description=(
        "Bilingual RAG-based medical triage API grounded in WHO and "
        "Pakistan Ministry of Health guidelines."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "X-API-Key"],
)


def verify_api_key(x_api_key: str = Header(default=None)) -> None:
    """
    Validate the X-API-Key header against API_ACCESS_KEY.

    API_ACCESS_KEY is required — the app refuses to start without it
    (see the check at import time above) — so this always enforces auth.

    Raises:
        HTTPException: 401 if the header is missing or incorrect.
    """
    if x_api_key != API_ACCESS_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check() -> HealthResponse:
    """Liveness check used by uptime monitors and load balancers."""
    return HealthResponse(status="ok")


@app.post(
    "/chat",
    response_model=ChatResponse,
    dependencies=[Depends(verify_api_key)],
    tags=["Chat"],
)
def chat(request: ChatRequest) -> ChatResponse:
    """
    Run the triage pipeline for a single message and return the result.

    Args:
        request: The user's message plus optional prior conversation
            history.

    Returns:
        The generated answer (or emergency alert), faithfulness grade,
        and cited sources.

    Raises:
        HTTPException: 500 if the underlying pipeline fails.
    """
    # Current message is appended to history before calling the
    # pipeline, matching app.py's behavior exactly (Streamlit appends
    # the current turn to session_state.messages before calling
    # process_user_message too) — keeps both entry points consistent.
    messages = [m.model_dump() for m in request.history]
    messages.append({"role": "user", "content": request.message})

    retriever = get_retriever()

    try:
        result = process_user_message(request.message, messages, retriever)
    except Exception:
        logger.exception("Pipeline failed for /chat request")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Something went wrong processing your message. Please try again.",
        )

    if result["is_emergency"]:
        return ChatResponse(is_emergency=True, answer=result["answer"])

    sources = [
        SourceDocument(content=doc.page_content, source=doc.metadata.get("source"))
        for doc in result["sources"]
    ]
    return ChatResponse(
        is_emergency=False,
        answer=result["answer"],
        grade=result["grade"],
        sources=sources,
    )