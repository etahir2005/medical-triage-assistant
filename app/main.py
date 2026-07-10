import sys
import os

sys.path.append(os.path.dirname(__file__))

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from chains import get_answer, translate_to_english
from emergency import check_emergency, get_emergency_response
from grader import grade_answer
from retriever import get_retriever


st.set_page_config(
    page_title="Medical Triage Assistant",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical Triage Assistant")
st.caption(
    "Bilingual health guidance based on WHO and Pakistan "
    "Ministry of Health guidelines"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

if "retriever" not in st.session_state:
    with st.spinner("Loading knowledge base..."):
        st.session_state.retriever = get_retriever()

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

user_input = st.chat_input(
    "Describe your symptoms in English, Urdu or Roman Urdu..."
)

if user_input:
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        english_input = translate_to_english(user_input)
    except Exception:
        english_input = user_input

    if check_emergency(user_input) or check_emergency(english_input):
        emergency_response = get_emergency_response()
        st.session_state.messages.append({
            "role": "assistant",
            "content": emergency_response
        })
        with st.chat_message("assistant"):
            st.error(emergency_response)

    else:
        with st.chat_message("assistant"):
            with st.spinner("Searching medical guidelines..."):
                try:
                    chat_history = []
                    for msg in st.session_state.messages[-6:]:
                        if msg["role"] == "user":
                            chat_history.append(
                                HumanMessage(content=msg["content"])
                            )
                        else:
                            chat_history.append(
                                AIMessage(content=msg["content"])
                            )

                    retrieved_docs = st.session_state.retriever.invoke(
                        english_input
                    )

                    answer, english_answer = get_answer(
                        user_input,
                        retrieved_docs,
                        chat_history
                    )

                    context = "\n\n".join([
                        doc.page_content for doc in retrieved_docs
                    ])

                    english_answer_graded = english_answer
                    grade = grade_answer(english_answer_graded, context)

                    st.markdown(answer)

                    if grade == "faithful":
                        st.success(
                            "✅ Faithful — answer grounded in guidelines"
                        )
                    elif grade == "partial":
                        st.warning(
                            "⚠️ Partially faithful — verify with a doctor"
                        )
                    else:
                        st.error(
                            "❌ Low confidence — please consult a "
                            "doctor directly"
                        )

                    st.divider()
                    st.caption(
                        "📄 Sources retrieved from knowledge base:"
                    )
                    for i, doc in enumerate(retrieved_docs):
                        with st.expander(f"Source {i+1}"):
                            st.write(doc.page_content)
                            source = doc.metadata.get("source", "")
                            if source:
                                st.caption(
                                    f"From: "
                                    f"{os.path.basename(source)}"
                                )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
                    })

                except Exception as e:
                    st.error(f"Error: {str(e)}")