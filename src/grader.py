"""Faithfulness grading for generated answers."""

import logging

from google.genai.errors import APIError
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import (
    GEMINI_API_KEY,
    GRADE_PARTIAL,
    GRADING_TEMPERATURE,
    LLM_MODEL_NAME,
    VALID_GRADES,
)
from src.utils import extract_text

logger = logging.getLogger(__name__)

_grading_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=GRADING_TEMPERATURE,
)

_grading_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an evaluator checking whether an AI
    generated answer is faithful to the provided context.

    Evaluate the answer against the context and respond with
    exactly one word only:

    - "faithful" if the answer is fully supported by the context
    - "partial" if the answer is partially supported but contains
      some information not in the context
    - "hallucinated" if the answer contains significant information
      not found in the context at all

    Respond with one word only. No explanation. No punctuation.

    CONTEXT:
    {context}

    ANSWER TO EVALUATE:
    {answer}
    """,
        ),
        ("human", "Evaluate the answer above."),
    ]
)


def grade_answer(answer: str, context: str) -> str:
    """
    Grade whether an answer is faithful to its source context.

    Args:
        answer: English version of the generated answer.
        context: English retrieved context used to generate it.

    Returns:
        One of "faithful", "partial", "hallucinated". Falls back to
        "partial" on unexpected model output or a failed API call.
    """
    try:
        chain = _grading_prompt | _grading_llm
        response = chain.invoke({"context": context, "answer": answer})
        grade = extract_text(response.content).strip().lower()
        return grade if grade in VALID_GRADES else GRADE_PARTIAL
    except APIError:
        logger.exception("Grading call failed; defaulting to partial")
        return GRADE_PARTIAL
