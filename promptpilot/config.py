import dspy
import os
from dotenv import load_dotenv

class AppConfig:
    """
    Manages application configuration, primarily loading from .env and setting up LLMs.
    """
    def __init__(self):
        load_dotenv()  # Load .env file from the project root
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.openai_model_name = os.getenv("OPENAI_MODEL_NAME", "gpt-3.5-turbo")
        # Future configurations can be added here:
        # self.another_api_key = os.getenv("ANOTHER_API_KEY")
        # self.default_temperature = float(os.getenv("DEFAULT_TEMPERATURE", 0.7))

    def get_configured_lm(self):
        """
        Initializes and returns a configured DSPy Language Model based on environment settings.
        Currently supports OpenAI.
        """
        if not self.openai_api_key:
            print("Warning: OPENAI_API_KEY not found in environment. OpenAI LLM will not be configured.")
            return None

        try:
            from dspy.openai import OpenAI
            # Parameters like model, api_key, max_tokens, temperature can be sourced from self
            llm = OpenAI(model=self.openai_model_name, api_key=self.openai_api_key)
            print(f"OpenAI LLM configured with model: {self.openai_model_name}")
            return llm
        except ImportError:
            print("Error: The 'openai' library is not installed. Please install it with 'pip install openai' to use the OpenAI LLM.")
            return None
        except Exception as e:
            print(f"Error configuring OpenAI LLM: {e}")
            return None

# Single global instance of AppConfig.
# This can be imported by other modules that need access to configuration.
# For more complex scenarios, dependency injection might be preferred over a global instance.
app_config = AppConfig()

def configure_dspy_globally():
    """
    Configures dspy.settings with the Language Model obtained from AppConfig.
    This should be called once at the application startup.
    """
    lm = app_config.get_configured_lm()
    if lm:
        dspy.settings.configure(lm=lm)
        print(f"DSPy globally configured with LM: {type(lm).__name__}")
        return True
    else:
        print("DSPy global LM configuration skipped: No Language Model could be initialized from AppConfig.")
        return False

if __name__ == '__main__':
    # Example of how to use this configuration module
    print("Attempting to configure DSPy globally...")
    if configure_dspy_globally():
        print("DSPy configured successfully.")
        if dspy.settings.lm:
            print(f"Current DSPy LM: {dspy.settings.lm}")
        else:
            print("DSPy LM is not set after configuration attempt.")
    else:
        print("DSPy configuration failed or was skipped.")

    # You can also access config values directly:
    # print(f"OpenAI Model configured: {app_config.openai_model_name}")
    # if not app_config.openai_api_key:
    #     print("OpenAI API Key is not set in the environment.")
