"""Pydantic request/response models for the HTTP API."""

from typing import Literal, Optional

from pydantic import BaseModel, Field

from src.config import MAX_INPUT_LENGTH


class ChatMessage(BaseModel):
    """A single message in a conversation history."""

    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    """Request body for POST /chat."""

    message: str = Field(
        ...,
        min_length=1,
        max_length=MAX_INPUT_LENGTH,
        description="Symptom description, in English, Roman Urdu, or Urdu script.",
    )
    history: list[ChatMessage] = Field(
        default_factory=list,
        description="Prior conversation turns, oldest first. Optional.",
    )


class SourceDocument(BaseModel):
    """A single retrieved source chunk cited in an answer."""

    content: str
    source: Optional[str] = None


class ChatResponse(BaseModel):
    """Response body for POST /chat."""

    is_emergency: bool
    answer: str
    grade: Optional[Literal["faithful", "partial", "hallucinated"]] = None
    sources: list[SourceDocument] = Field(default_factory=list)


class HealthResponse(BaseModel):
    """Response body for GET /health."""

    status: Literal["ok"]
