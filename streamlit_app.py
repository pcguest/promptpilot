"""
A simple Streamlit application template for PromptPilot.

This app provides a basic User Interface to interact with the SmartAnswerModule.

To run this Streamlit app:
1. Ensure you have all necessary dependencies:
   pip install streamlit
   (And ensure promptpilot itself is runnable, e.g., by being in the project root
    or having promptpilot installed.)
2. Configure your .env file with the required API keys (e.g., OPENAI_API_KEY).
3. Run from the project root directory:
   streamlit run streamlit_app.py
"""
import streamlit as st
import sys
import os
import logging

# --- Path Setup ---
# Ensure the package root is in sys.path for direct script execution
# and for Streamlit to find the promptpilot modules.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = SCRIPT_DIR # If streamlit_app.py is in the root
# If streamlit_app.py is in a subdirectory like 'ui/', adjust PROJECT_ROOT:
# PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# --- Application Imports ---
# It's important that these imports happen *after* sys.path is configured.
try:
    from promptpilot.app import configure_dspy_globally # For DSPy setup
    from promptpilot.modules.smart_answer import SmartAnswerModule
    import dspy
except ImportError as e:
    st.error(f"Error importing PromptPilot modules: {e}. "
             "Ensure the app is run from the project root or PYTHONPATH is set correctly.")
    # Stop further execution if core modules can't be imported.
    st.stop()

# --- Logger Setup ---
# Using a basic logger. For more advanced logging, configure as needed.
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

# --- Streamlit App UI and Logic ---

st.set_page_config(page_title="PromptPilot Smart QA", layout="centred")

st.title("🤖 PromptPilot - Smart Question Answering")

st.markdown("""
Welcome to the PromptPilot Smart QA demo!
This application uses the `SmartAnswerModule` powered by DSPy to answer your questions.
""")

# --- DSPy Configuration ---
# Memoize the DSPy configuration to avoid re-running on every interaction.
@st.cache_resource
def initialise_dspy():
    """Initialises DSPy settings globally."""
    if not configure_dspy_globally():
        st.error("DSPy LM configuration failed. Please check your .env file and API keys.")
        return None

    if not dspy.settings.lm:
        st.warning("DSPy LM is not configured. Answers may not be generated.")
        return None

    logger.info("DSPy initialised successfully for Streamlit app.")
    return dspy.settings.lm

# --- SmartAnswerModule Instantiation ---
# Memoize the module instantiation.
@st.cache_resource
def get_smart_answer_module():
    """Gets an instance of the SmartAnswerModule."""
    try:
        module = SmartAnswerModule()
        logger.info("SmartAnswerModule initialised successfully.")
        return module
    except dspy.DSPyError as e:
        st.error(f"Error initialising SmartAnswerModule: {e}. This usually means the LLM is not configured.")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred while initialising SmartAnswerModule: {e}")
        logger.error("Unexpected error during SmartAnswerModule init", exc_info=True)
        return None

# Initialize DSPy and the module
# This will run once and be cached.
llm_configured = initialize_dspy()
smart_answer_pipeline = None
if llm_configured:
    smart_answer_pipeline = get_smart_answer_module()

# --- Main Interaction ---
st.subheader("Ask a Question")

with st.form("qa_form"):
    user_question = st.text_area("Enter your question here:", height=100, key="user_question")
    submit_button = st.form_submit_button("Get Answer")

if submit_button and user_question:
    if not llm_configured:
        st.error("Cannot process question: Language Model is not configured.")
    elif not smart_answer_pipeline:
        st.error("Cannot process question: SmartAnswerModule is not available.")
    else:
        st.markdown("---")
        st.write(f"❓ **You asked:** *{user_question}*")
        try:
            with st.spinner("Thinking..."):
                # Ensure the module's forward method is called correctly
                response = smart_answer_pipeline(question=user_question)

            st.write("💡 **Answer:**")
            st.info(response.answer if response and hasattr(response, 'answer') else "No answer received.")

            # Optionally show DSPy history if desired (for debugging)
            # if st.checkbox("Show DSPy trace (last interaction)"):
            #     st.text_area("DSPy Trace:", value=str(dspy.settings.lm.history[-1]), height=300)

        except dspy.DSPyError as e:
            st.error(f"DSPy Error: {e}")
            logger.error(f"DSPyError during question answering: {e}", exc_info=True)
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            logger.error(f"Unexpected error during question answering: {e}", exc_info=True)
elif submit_button and not user_question:
    st.warning("Please enter a question.")

st.markdown("---")
st.caption("Powered by PromptPilot and DSPy.")

# --- Instructions for Running ---
with st.expander("How to Run This App"):
    st.markdown("""
    1.  **Save this code** as `streamlit_app.py` in the root of your PromptPilot project.
    2.  **Install Streamlit**: `pip install streamlit`
    3.  **Ensure your `.env` file is correctly configured** with your LLM API keys (e.g., `OPENAI_API_KEY`).
    4.  **Open your terminal, navigate to the project root directory**, and run:
        ```bash
        streamlit run streamlit_app.py
        ```
    """)

logger.info("Streamlit app script execution finished (reached end of script).")
