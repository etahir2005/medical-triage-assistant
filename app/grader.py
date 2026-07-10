import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"


def extract_text(content) -> str:
    """
    Handles both string and list response formats from Gemini.
    """
    if isinstance(content, list):
        return "".join([
            block.get("text", "") if isinstance(block, dict)
            else str(block)
            for block in content
        ])
    return str(content)


def grade_answer(answer: str, context: str) -> str:
    """
    Call 3 — Grades whether the answer is faithful to the context.
    Both answer and context are in English for accurate comparison.
    Returns: faithful / partial / hallucinated
    """
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an evaluator checking whether an AI
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
        """),
        ("human", "Evaluate the answer above.")
    ])

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "answer": answer
    })

    grade = extract_text(response.content).strip().lower()

    if grade not in ["faithful", "partial", "hallucinated"]:
        return "partial"

    return grade