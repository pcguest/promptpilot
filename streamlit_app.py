"""Streamlit web application for the PromptPilot project.

This module provides a simple web interface for interacting with the PromptPilot
agents. It is built with Streamlit and is designed to be a user-friendly way
to demonstrate the capabilities of the application.
"""

import streamlit as st

from src.promptpilot.components.smart_answer_agent import SmartAnswerAgent
from src.promptpilot.config import configure_dspy

# --- Page Configuration ---
st.set_page_config(
    page_title="PromptPilot",
    page_icon="🧠",
    layout="centered",
)

# --- LLM Configuration ---
try:
    configure_dspy()
except (ValueError, ImportError, RuntimeError) as e:
    st.error(str(e))
    st.stop()

# --- Application ---
st.title("🧠 PromptPilot")
st.markdown("A discerning answering agent, powered by DSPy.")


# --- DSPy Module ---
@st.cache_resource
def load_smart_answer_agent():
    """Load the DSPy module, cached for performance."""
    return SmartAnswerAgent()


smart_answer_agent = load_smart_answer_agent()

# --- User Interaction ---
with st.container():
    st.write("Kindly pose your query below:")
    user_question = st.text_input(
        "Your question:", key="question_input", label_visibility="collapsed"
    )

    if st.button("Obtain Answer", key="get_answer_button"):
        if user_question:
            with st.spinner("Just a moment, formulating a response..."):
                try:
                    result = smart_answer_agent.forward(question=user_question)
                    st.info(result)  # Changed from st.success to st.info
                    # for a more neutral tone
                except Exception as e:
                    st.error(f"An unforeseen issue occurred: {e}")
        else:
            st.warning("Please do type in a question before proceeding.")
