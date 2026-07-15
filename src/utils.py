"""Shared helper functions used across the pipeline."""

import re

_URDU_SCRIPT_PATTERN = re.compile(r"[\u0600-\u06FF\u0750-\u077F]")


def contains_urdu_script(text: str) -> bool:
    """
    Deterministically check whether text contains Urdu/Arabic script
    characters, using Unicode range detection rather than LLM judgment.

    Args:
        text: Text to check.

    Returns:
        True if any character falls in the Arabic/Urdu Unicode blocks.
    """
    return bool(_URDU_SCRIPT_PATTERN.search(text or ""))


def extract_text(content) -> str:
    """
    Normalize a Gemini response's `.content` into a plain string.

    Args:
        content: The `.content` attribute of a LangChain AIMessage —
            either a string or a list of dict/str blocks.

    Returns:
        The extracted plain-text string.
    """
    if isinstance(content, list):
        return "".join(
            block.get("text", "") if isinstance(block, dict) else str(block)
            for block in content
        )
    return str(content)
