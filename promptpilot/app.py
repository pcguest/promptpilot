import dspy # Still needed for dspy.settings.lm check and dspy.settings.inspect_history
import sys
import os
import logging

# Ensure the package root is in sys.path for direct script execution.
# This allows 'from promptpilot.modules...' to work when running 'python promptpilot/app.py'.
# This needs to be done *before* other promptpilot imports.
if __package__ is None or __package__ == '':
    script_dir = os.path.dirname(os.path.abspath(__file__)) # promptpilot/app.py
    project_root = os.path.dirname(script_dir) # Parent directory of promptpilot/
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Import from our refactored config and modules
from promptpilot.config import configure_dspy_globally
from promptpilot.modules.smart_answer import SmartAnswerModule

# --- Application Logic ---
def run_smart_answer_example():
    """
    Demonstrates the usage of the SmartAnswerModule.
    Assumes DSPy has been configured globally.
    """
    if not dspy.settings.lm:
        logging.warning("\nSkipping SmartAnswerModule example: DSPy LM is not configured globally.")
        logging.warning("Please ensure your .env file is set up and OPENAI_API_KEY is valid.")
        return

    logging.info("\n--- Running SmartAnswerModule Example ---")
    try:
        # SmartAnswerModule will raise an error if LM is not configured,
        # but we check dspy.settings.lm first for a clearer message.
        smart_answer_pipeline = SmartAnswerModule()

        question = "What is the primary benefit of using DSPy for prompt engineering?"
        logging.info(f"Asking: \"{question}\"")

        response = smart_answer_pipeline(question=question)
        logging.info(f"Answer: {response.answer}")

        # To inspect the last interaction (prompt, response, etc.) with the LM:
        # dspy.settings.inspect_history(n=1)
        # Note: This requires the LM to have history tracking enabled (default for many).

    except dspy.DSPyError as e:
        logging.error(f"A DSPyError occurred while running SmartAnswerModule: {e}")
        logging.error("This could be due to issues with the LLM provider, API keys, or network.")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}", exc_info=True)

def main():
    """
    Main function to initialise configurations and run the PromptPilot application.
    """
    # Basic logging configuration
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    # You can create specific loggers if needed, e.g., logger = logging.getLogger(__name__)
    # For this simple app, the root logger is fine.

    logging.info("Welcome to PromptPilot!")

    # Configure DSPy settings globally using the new config module.
    # This will attempt to load .env, find API keys, and set dspy.settings.lm.
    if not configure_dspy_globally():
        # The configure_dspy_globally function already uses logging.warning/error.
        logging.warning("Application might not function as expected due to LM configuration issues.")
        # Depending on the application, you might want to exit or allow partial functionality.
        # For this example, we'll still try to proceed but run_smart_answer_example has its own checks.

    # Run example application logic
    run_smart_answer_example()

    logging.info("\nPromptPilot execution finished.")
    # Add more application logic or calls to other modules here as needed.

if __name__ == "__main__":
    # The `if __package__ ...` block at the top handles path adjustments
    # for direct script execution.
    main()
