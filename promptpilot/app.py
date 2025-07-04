import dspy # Still needed for dspy.settings.lm check and dspy.settings.inspect_history
import sys
import os

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
        print("\nSkipping SmartAnswerModule example: DSPy LM is not configured globally.")
        print("Please ensure your .env file is set up and OPENAI_API_KEY is valid.")
        return

    print("\n--- Running SmartAnswerModule Example ---")
    try:
        # SmartAnswerModule will raise an error if LM is not configured,
        # but we check dspy.settings.lm first for a clearer message.
        smart_answer_pipeline = SmartAnswerModule()

        question = "What is the primary benefit of using DSPy for prompt engineering?"
        print(f"Asking: \"{question}\"")

        response = smart_answer_pipeline(question=question)
        print(f"Answer: {response.answer}")

        # To inspect the last interaction (prompt, response, etc.) with the LM:
        # dspy.settings.inspect_history(n=1)
        # Note: This requires the LM to have history tracking enabled (default for many).

    except dspy.DSPyError as e:
        print(f"A DSPyError occurred while running SmartAnswerModule: {e}")
        print("This could be due to issues with the LLM provider, API keys, or network.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")

def main():
    """
    Main function to initialize configurations and run the PromptPilot application.
    """
    print("Welcome to PromptPilot!")

    # Configure DSPy settings globally using the new config module.
    # This will attempt to load .env, find API keys, and set dspy.settings.lm.
    if not configure_dspy_globally():
        # The configure_dspy_globally function already prints detailed messages.
        print("Application might not function as expected due to LM configuration issues.")
        # Depending on the application, you might want to exit or allow partial functionality.
        # For this example, we'll still try to proceed but run_smart_answer_example has its own checks.

    # Run example application logic
    run_smart_answer_example()

    print("\nPromptPilot execution finished.")
    # Add more application logic or calls to other modules here as needed.

if __name__ == "__main__":
    # The `if __package__ ...` block at the top handles path adjustments
    # for direct script execution.
    main()
