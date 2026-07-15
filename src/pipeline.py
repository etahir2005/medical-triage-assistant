"""Orchestrates translation, emergency check, retrieval, generation, grading."""

import logging

from google.genai.errors import APIError
from langchain_core.messages import AIMessage, HumanMessage
from pinecone.exceptions import PineconeException

from src.chains import detect_language_and_translate, get_answer
from src.config import CHAT_HISTORY_LIMIT, LANGUAGE_ENGLISH
from src.emergency import check_emergency, get_emergency_response
from src.grader import grade_answer

logger = logging.getLogger(__name__)


def build_chat_history(messages: list) -> list:
    """
    Convert session-state message dicts into LangChain message objects.

    Args:
        messages: List of {"role": ..., "content": ...} dicts.

    Returns:
        The last CHAT_HISTORY_LIMIT messages as HumanMessage/AIMessage
        objects, in order.
    """
    history = []
    for msg in messages[-CHAT_HISTORY_LIMIT:]:
        cls = HumanMessage if msg["role"] == "user" else AIMessage
        history.append(cls(content=msg["content"]))
    return history


def process_user_message(user_input: str, messages: list, retriever) -> dict:
    """
    Run the full triage pipeline for a single user message.

    Each stage is wrapped separately so failures are logged with a
    stage-specific message, making production issues easier to
    diagnose from logs alone.

    Args:
        user_input: The raw text the user submitted.
        messages: Full session chat history (list of role/content dicts).
        retriever: The cached Pinecone retriever.

    Returns:
        dict with key "is_emergency", plus "answer" always, and
        "grade" / "sources" when not an emergency.

    Raises:
        PineconeException: Propagates retrieval failures to the
            caller after logging which stage failed.
        APIError: Propagates Gemini generation failures to the
            caller after logging which stage failed.
    """
    try:
        response_language, english_input = detect_language_and_translate(user_input)
    except APIError:
        logger.warning("Translation failed, using original text", exc_info=True)
        response_language, english_input = LANGUAGE_ENGLISH, user_input

    if check_emergency(user_input) or check_emergency(english_input):
        return {"is_emergency": True, "answer": get_emergency_response()}

    chat_history = build_chat_history(messages)

    try:
        retrieved_docs = retriever.invoke(english_input)
    except PineconeException:
        logger.exception("Retrieval failed")
        raise

    try:
        answer, english_answer = get_answer(
            user_input, retrieved_docs, chat_history, response_language
        )
    except APIError:
        logger.exception("Answer generation failed")
        raise

    context = "\n\n".join(doc.page_content for doc in retrieved_docs)
    grade = grade_answer(english_answer, context)

    return {
        "is_emergency": False,
        "answer": answer,
        "grade": grade,
        "sources": retrieved_docs,
    }
