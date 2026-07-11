"""LLM chains for translation and bilingual answer generation."""

import logging

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import (
    ANSWER_TEMPERATURE,
    GEMINI_API_KEY,
    LANGUAGE_ENGLISH,
    LANGUAGE_ROMAN_URDU,
    LANGUAGE_URDU_SCRIPT,
    LLM_MODEL_NAME,
    TRANSLATION_TEMPERATURE,
    VALID_LANGUAGES,
)
from src.utils import contains_urdu_script, extract_text

logger = logging.getLogger(__name__)

_translation_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=TRANSLATION_TEMPERATURE,
)
_answer_llm = ChatGoogleGenerativeAI(
    model=LLM_MODEL_NAME,
    google_api_key=GEMINI_API_KEY,
    temperature=ANSWER_TEMPERATURE,
)

LANGUAGE_INSTRUCTIONS = {
    LANGUAGE_ENGLISH: (
        "You MUST respond entirely in English. Do not use Urdu or Roman Urdu."
    ),
    LANGUAGE_ROMAN_URDU: (
        "You MUST respond entirely in Roman Urdu (Urdu words spelled "
        'using English letters, e.g. "aap ko bukhaar hai"). Do not use '
        "English or Urdu script."
    ),
    LANGUAGE_URDU_SCRIPT: (
        "You MUST respond entirely in Urdu script (اردو رسم الخط). "
        "Do not use English or Roman Urdu."
    ),
}

SYSTEM_PROMPT = """You are a medical triage assistant helping people
in Pakistan understand their symptoms and decide what to do next.

LANGUAGE INSTRUCTION FOR THIS RESPONSE (follow this exactly, it has
already been determined from the user's current question and does
not depend on conversation history):
{language_instruction}

IMPORTANT RULES:
- Answer ONLY based on the context provided below
- If the answer is not in the context, say (translated into the
  required response language):
  "I could not find reliable information about this in my
  knowledge base. Please consult a doctor directly."
- Never diagnose — only provide triage guidance
- Keep answers clear and simple
- Always recommend seeing a doctor for serious symptoms
- Use the conversation history only to understand what the current
  question is about (e.g. follow-up questions). The history's
  language must NOT influence your response language — only the
  LANGUAGE INSTRUCTION above does.
- End every response with this disclaimer, translated into the
  required response language:
  "This is not medical advice. Please consult a qualified
  healthcare professional for proper diagnosis and treatment."

VERY IMPORTANT — After the disclaimer, add exactly this on a
new line with no extra text before it:
ENGLISH_VERSION: [write the complete answer in English here]

The ENGLISH_VERSION is used internally for quality checking
and will NOT be shown to the user.

CONTEXT FROM MEDICAL GUIDELINES:
{context}

REMINDER: {language_instruction}
"""

_answer_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", SYSTEM_PROMPT),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)


def detect_language_and_translate(text: str) -> tuple:
    """
    Detect the input language and translate to English in one call.

    Urdu script is detected deterministically via Unicode range
    checking rather than LLM judgment, which is 100% reliable. The
    LLM is only used to distinguish English from Roman Urdu, since
    both use Latin script.

    Args:
        text: User-supplied text in English, Urdu, or Roman Urdu.

    Returns:
        Tuple of (language_label, english_text), where language_label
        is one of "english", "roman_urdu", "urdu_script".

    Raises:
        Exception: Propagates any LLM call failure to the caller.
    """
    if contains_urdu_script(text):
        response = _translation_llm.invoke(
            f"Translate this Urdu text to English. Return only the "
            f"translation, nothing else: {text}"
        )
        return LANGUAGE_URDU_SCRIPT, extract_text(response.content).strip()

    response = _translation_llm.invoke(
        "Analyze the text below and respond in exactly this format, "
        "nothing else:\n"
        "LANGUAGE: <english or roman_urdu>\n"
        "TRANSLATION: <the English translation, or the original text "
        "if it is already English>\n\n"
        f"Text: {text}"
    )
    raw = extract_text(response.content).strip()

    language = LANGUAGE_ENGLISH
    translation = text
    if "LANGUAGE:" in raw and "TRANSLATION:" in raw:
        lang_part = (
            raw.split("LANGUAGE:", 1)[1].split("TRANSLATION:", 1)[0].strip().lower()
        )
        translation_part = raw.split("TRANSLATION:", 1)[1].strip()
        if lang_part in (LANGUAGE_ENGLISH, LANGUAGE_ROMAN_URDU):
            language = lang_part
        if translation_part:
            translation = translation_part
    else:
        logger.warning("Language detection response malformed: %r", raw)

    return language, translation


def _build_context(retrieved_docs: list) -> str:
    """Format retrieved chunks into a numbered context block."""
    return "\n\n".join(
        f"Source {i + 1}:\n{doc.page_content}" for i, doc in enumerate(retrieved_docs)
    )


def _parse_bilingual_response(full_response: str) -> tuple:
    """Split the raw LLM output into (user_answer, english_answer)."""
    if "ENGLISH_VERSION:" in full_response:
        user_answer, english_answer = full_response.split("ENGLISH_VERSION:", 1)
        return user_answer.strip(), english_answer.strip()
    logger.warning("Model omitted ENGLISH_VERSION marker; grading may be inaccurate")
    return full_response.strip(), full_response.strip()


def get_answer(
    question: str,
    retrieved_docs: list,
    chat_history: list = None,
    response_language: str = LANGUAGE_ENGLISH,
) -> tuple:
    """
    Generate a bilingual answer grounded in retrieved context.

    Args:
        question: The user's original-language question.
        retrieved_docs: Chunks retrieved from Pinecone.
        chat_history: Prior LangChain message objects for context.
        response_language: One of "english", "roman_urdu",
            "urdu_script" — determined once via
            detect_language_and_translate() and enforced as a hard
            constraint here, rather than re-decided by this call.
            Falls back to English if not a recognized value.

    Returns:
        Tuple of (user_facing_answer, english_answer_for_grading).
    """
    chat_history = chat_history or []
    context = _build_context(retrieved_docs)

    if response_language not in VALID_LANGUAGES:
        logger.warning(
            "Unknown response_language %r, defaulting to English", response_language
        )
        response_language = LANGUAGE_ENGLISH

    language_instruction = LANGUAGE_INSTRUCTIONS[response_language]

    chain = _answer_prompt | _answer_llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
            "chat_history": chat_history,
            "language_instruction": language_instruction,
        }
    )

    full_response = extract_text(response.content)
    return _parse_bilingual_response(full_response)
