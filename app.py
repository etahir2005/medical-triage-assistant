"""Streamlit entry point for the Medical Triage Assistant."""

import logging
import os

import streamlit as st

from src.config import MAX_INPUT_LENGTH
from src.pipeline import process_user_message
from src.retriever import get_retriever

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


def render_result(result: dict) -> None:
    """
    Render a pipeline result to the chat UI and save it to session state.

    Args:
        result: The dict returned by process_user_message().
    """
    if result["is_emergency"]:
        st.error(result["answer"])
        st.session_state.messages.append(
            {"role": "assistant", "content": result["answer"]}
        )
        return

    st.markdown(result["answer"])

    grade = result["grade"]
    if grade == "faithful":
        st.success("✅ Faithful — answer grounded in guidelines")
    elif grade == "partial":
        st.warning("⚠️ Partially faithful — verify with a doctor")
    else:
        st.error("❌ Low confidence — please consult a doctor directly")

    st.divider()
    st.caption("📄 Sources retrieved from knowledge base:")
    for i, doc in enumerate(result["sources"]):
        with st.expander(f"Source {i + 1}"):
            st.write(doc.page_content)
            source = doc.metadata.get("source", "")
            if source:
                st.caption(f"From: {os.path.basename(source)}")

    st.session_state.messages.append({"role": "assistant", "content": result["answer"]})


st.set_page_config(
    page_title="Medical Triage Assistant",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 Medical Triage Assistant")
st.caption(
    "Bilingual health guidance based on WHO and Pakistan "
    "Ministry of Health guidelines"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

with st.spinner("Loading knowledge base..."):
    retriever = get_retriever()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input("Describe your symptoms in English, Urdu or Roman Urdu...")

if user_input:
    if len(user_input) > MAX_INPUT_LENGTH:
        st.warning(f"Please limit your message to {MAX_INPUT_LENGTH} characters.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Searching medical guidelines..."):
                try:
                    result = process_user_message(
                        user_input, st.session_state.messages, retriever
                    )
                except Exception:
                    logger.exception("Pipeline failed")
                    st.error(
                        "Something went wrong processing your message. "
                        "Please try again."
                    )
                    result = None

            if result:
                render_result(result)
