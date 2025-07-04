import os
import dspy
from dotenv import load_dotenv

# Ensure the package root is in sys.path for direct script execution.
# This allows 'from promptpilot.modules...' to work when running 'python promptpilot/app.py'.
if __package__ is None or __package__ == '':
    # Current script directory (e.g., /path/to/promptpilot)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Project root (e.g., /path/to/)
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from promptpilot.modules.smart_answer import SmartAnswerModule

# Load .env file if it exists, for OPENAI_API_KEY and other environment variables
load_dotenv()

def configure_dspy_settings():
    """
    Configures DSPy settings. Uses OpenAI if API key is found,
    otherwise falls back to a DummyLM for keyless operation.
    """
    openai_api_key = os.getenv("OPENAI_API_KEY")
    openai_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo") # Default model

    if openai_api_key:
        print(f"OpenAI API key found. Configuring DSPy with OpenAI model: {openai_model_name}")
        # For dspy-ai 3.x, dspy.OpenAI is the way to configure the LM
        llm = dspy.OpenAI(model=openai_model_name, api_key=openai_api_key)
        dspy.settings.configure(lm=llm)
    else:
        print("OpenAI API key not found. Configuring DSPy with a DummyLM.")

        # Define a simple DummyLM for keyless operation. Must inherit from dspy.LM.
        class AppDummyLM(dspy.LM):
            def __init__(self, model_name="dummy-app-model"): # LMs usually take a model name
                super().__init__(model_name) # Pass model_name to super
                self.provider = "dummy" # Standard LM attribute

            def basic_request(self, prompt, **kwargs):
                # This DummyLM's basic_request needs to return a JSON parsable string
                # if SmartAnswerModule's Predictor expects JSON from the LM.
                # Based on previous test failures, JSON output is expected.
                response_text = "This is a dummy answer from AppDummyLM for prompt: " + prompt[:50] + "..."
                # Assuming the signature expects an 'answer' field.
                json_output = json.dumps({"answer": response_text})
                return [json_output]

            def __call__(self, *args, **kwargs):
                # Similar to basic_request, ensure output is parseable by adapters.
                prompt_content_for_logic = "No prompt identified in __call__"
                if args and isinstance(args[0], str):
                     prompt_content_for_logic = args[0]
                elif "prompt" in kwargs:
                     prompt_content_for_logic = kwargs["prompt"]
                elif "messages" in kwargs and kwargs["messages"]:
                     prompt_content_for_logic = kwargs["messages"][-1].get("content", "")

                response_text = "This is a dummy answer from AppDummyLM __call__ for: " + prompt_content_for_logic[:50] + "..."
                json_output = json.dumps({"answer": response_text})
                return [json_output]

        # Need to import json if AppDummyLM is producing JSON
        import json
        dspy.settings.configure(lm=AppDummyLM())

def run_application():
    """
    Main application logic.
    """
    print("\nWelcome to PromptPilot!")

    configure_dspy_settings() # Configure DSPy with OpenAI or DummyLM

    print("\n--- Running SmartAnswerModule Example ---")
    try:
        smart_answer_pipeline = SmartAnswerModule()
        question = "What is the primary benefit of using DSPy for prompt engineering?"
        print(f"Asking: \"{question}\"")

        # Use the forward method as suggested for module interaction
        response = smart_answer_pipeline.forward(question=question)
        # dspy.Prediction objects have attributes corresponding to OutputFields
        print(f"Answer: {response.answer}")

    except Exception as e: # Catching generic Exception for now
        print(f"An error occurred whilst running the SmartAnswerModule: {e}")
        print("This could be due to issues with the LLM provider, API keys, network, or internal logic.")

    print("\nPromptPilot execution finished.")

if __name__ == "__main__":
    run_application()
