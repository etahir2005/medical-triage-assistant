import os

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI


load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MODEL_NAME = "gemini-3.1-flash-lite"


def extract_text(content) -> str:
    """
    Handles both string and list response formats from Gemini.
    Newer versions of langchain_google_genai return content as a list
    of blocks rather than a plain string.
    """
    if isinstance(content, list):
        return "".join([
            block.get("text", "") if isinstance(block, dict)
            else str(block)
            for block in content
        ])
    return str(content)


def translate_to_english(text: str) -> str:
    """
    Call 1 — Translates any language to English.
    Used for emergency check and Pinecone retrieval.
    If text is already English returns it unchanged.
    """
    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0
    )
    response = llm.invoke(
        f"Translate this to English. If it is already in English "
        f"return it as is. Return only the translation, "
        f"nothing else: {text}"
    )
    return extract_text(response.content).strip()


def get_answer(question: str, retrieved_docs: list,
               chat_history: list = None) -> tuple:
    """
    Call 2 — Generates answer in user's language.
    Also returns an English version inside the same response
    so we avoid a separate translation call for grading.
    Returns (user_answer, english_answer) as a tuple.
    """
    if chat_history is None:
        chat_history = []

    context = "\n\n".join([
        f"Source {i+1}:\n{doc.page_content}"
        for i, doc in enumerate(retrieved_docs)
    ])

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a medical triage assistant helping people
        in Pakistan understand their symptoms and decide what to do next.

        IMPORTANT RULES:
        - Answer ONLY based on the context provided below
        - If the answer is not in the context, say:
          "I could not find reliable information about this in my
          knowledge base. Please consult a doctor directly."
        - Never diagnose — only provide triage guidance
        - Keep answers clear and simple
        - Always recommend seeing a doctor for serious symptoms
        - Always respond in the same language the user used.
          If they wrote in Urdu script, respond in Urdu script.
          If they wrote in Roman Urdu (e.g. "mujhe bukhaar hai"),
          respond in Roman Urdu.
          If they wrote in English, respond in English.
        - Use the conversation history to understand follow up questions
        - End every response with this disclaimer:
          "This is not medical advice. Please consult a qualified
          healthcare professional for proper diagnosis and treatment."

        VERY IMPORTANT — After the disclaimer, add exactly this on a
        new line with no extra text before it:
        ENGLISH_VERSION: [write the complete answer in English here]

        The ENGLISH_VERSION is used internally for quality checking
        and will NOT be shown to the user.

        CONTEXT FROM MEDICAL GUIDELINES:
        {context}
        """),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])

    llm = ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=GEMINI_API_KEY,
        temperature=0.1
    )

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question,
        "chat_history": chat_history
    })

    full_response = extract_text(response.content)

    # Split response into user answer and English version for grading
    if "ENGLISH_VERSION:" in full_response:
        parts = full_response.split("ENGLISH_VERSION:")
        user_answer = parts[0].strip()
        english_answer = parts[1].strip()
    else:
        # Fallback if model does not include the marker
        user_answer = full_response
        english_answer = full_response

    return user_answer, english_answer