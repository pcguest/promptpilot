import dspy
import os
from dotenv import load_dotenv
import sys

# Ensure the package root is in sys.path for direct script execution
# This allows 'from promptpilot.modules...' to work when running 'python promptpilot/app.py'
if __package__ is None or __package__ == '':
    # Get the directory of the current script (app.py)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Get the parent directory (project root, assuming app.py is in promptpilot/)
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

# Import your DSPy modules
from promptpilot.modules.smart_answer import SmartAnswerModule


# --- Configuration ---
def configure_dspy_settings():
    """
    Configures DSPy settings, primarily the Language Model.
    Tries to load settings from environment variables.
    """
    load_dotenv()  # Load .env file if present

    # Example: Configure for OpenAI
    # Ensure you have OPENAI_API_KEY in your .env file or environment
    api_key = os.getenv("OPENAI_API_KEY")
    model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo") # Default model

    if api_key:
        try:
            from dspy.openai import OpenAI
            llm = OpenAI(model=model_name, api_key=api_key)
            dspy.settings.configure(lm=llm)
            print(f"DSPy configured with OpenAI model: {model_name}")
        except ImportError:
            print("OpenAI library not found. Please install it: pip install openai")
        except Exception as e:
            print(f"Error configuring OpenAI: {e}")
    else:
        print("OPENAI_API_KEY not found in environment variables.")
        print("DSPy LM is not configured. Some functionalities will not work.")
        print("Please create a .env file with OPENAI_API_KEY='your-key' or set it as an environment variable.")
        print("Alternatively, configure a different LLM provider in `configure_dspy_settings` in app.py.")

# --- Application Logic ---
def run_smart_answer_example():
    """
    Demonstrates the usage of the SmartAnswerModule.
    """
    if not dspy.settings.lm:
        print("\nSkipping SmartAnswerModule example because DSPy LM is not configured.")
        return

    print("\n--- Running SmartAnswerModule Example ---")
    try:
        smart_answer_pipeline = SmartAnswerModule()
        question = "What is the capital of Australia?"
        print(f"Asking: \"{question}\"")
        response = smart_answer_pipeline(question=question)
        print(f"Answer: {response.answer}")

        # You can inspect the last interaction with dspy.settings.inspect_history()
        # dspy.settings.inspect_history(n=1)

    except Exception as e:
        print(f"Error running SmartAnswerModule: {e}")
        print("This might be due to an issue with the LLM configuration or API request.")

def main():
    """
    Main function to run the PromptPilot application.
    """
    print("Welcome to PromptPilot!")

    # Configure DSPy settings (e.g., LLM)
    configure_dspy_settings()

    # Run examples or main application logic
    run_smart_answer_example()

    # Add more application logic here as needed

if __name__ == "__main__":
    main()
