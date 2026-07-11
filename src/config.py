"""Centralized configuration and constants for the Medical Triage Assistant."""

import os

from dotenv import load_dotenv

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

if not GEMINI_API_KEY:
    raise EnvironmentError("GEMINI_API_KEY is not set. Add it to your .env file.")
if not PINECONE_API_KEY:
    raise EnvironmentError("PINECONE_API_KEY is not set. Add it to your .env file.")

LLM_MODEL_NAME = "gemini-3.1-flash-lite"
EMBEDDING_MODEL_NAME = "BAAI/bge-base-en-v1.5"

PINECONE_INDEX_NAME = "medical-triage"
EMBEDDING_DIMENSION = 768
RETRIEVAL_TOP_K = 5

CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
INGESTION_BATCH_SIZE = 100

CHAT_HISTORY_LIMIT = 6
MAX_INPUT_LENGTH = 1000

TRANSLATION_TEMPERATURE = 0
ANSWER_TEMPERATURE = 0  # lowered from 0.1 for consistent language-matching
GRADING_TEMPERATURE = 0

GRADE_FAITHFUL = "faithful"
GRADE_PARTIAL = "partial"
GRADE_HALLUCINATED = "hallucinated"
VALID_GRADES = {GRADE_FAITHFUL, GRADE_PARTIAL, GRADE_HALLUCINATED}

LANGUAGE_ENGLISH = "english"
LANGUAGE_ROMAN_URDU = "roman_urdu"
LANGUAGE_URDU_SCRIPT = "urdu_script"
VALID_LANGUAGES = {LANGUAGE_ENGLISH, LANGUAGE_ROMAN_URDU, LANGUAGE_URDU_SCRIPT}
