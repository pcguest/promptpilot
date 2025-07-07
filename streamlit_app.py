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
st.markdown("A smart answering agent powered by DSPy.")


# --- DSPy Module ---
@st.cache_resource
def load_smart_answer_agent():
    """Load the DSPy module, cached for performance."""
    return SmartAnswerAgent()


smart_answer_agent = load_smart_answer_agent()

# --- User Interaction ---
user_question = st.text_input("Ask a question:", key="question_input")

if st.button("Get Answer", key="get_answer_button"):
    if user_question:
        with st.spinner("Thinking..."):
            try:
                result = smart_answer_agent.forward(question=user_question)
                st.success(result)
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.warning("Please enter a question.")
