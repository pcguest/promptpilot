import dspy
import os
from dotenv import load_dotenv
import logging

# It's good practice for libraries/modules to use getLogger(__name__)
# instead of basicConfig, which is usually for applications.
# However, since this config is tightly coupled with the app's startup,
# messages here are part of the app's initialisation process.
# We will use a logger instance.
logger = logging.getLogger(__name__)

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
        Initialises and returns a configured DSPy Language Model based on environment settings.
        Currently supports OpenAI.
        """
        if not self.openai_api_key:
            logger.warning("OPENAI_API_KEY not found in environment. OpenAI LLM will not be configured.")
            return None

        try:
            from dspy.openai import OpenAI
            # Parameters like model, api_key, max_tokens, temperature can be sourced from self
            llm = OpenAI(model=self.openai_model_name, api_key=self.openai_api_key)
            logger.info(f"OpenAI LLM configured with model: {self.openai_model_name}")
            return llm
        except ImportError:
            logger.error("The 'openai' library is not installed. Please install it with 'pip install openai' to use the OpenAI LLM.")
            return None
        except Exception as e:
            logger.error(f"Error configuring OpenAI LLM: {e}", exc_info=True)
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
        logger.info(f"DSPy globally configured with LM: {type(lm).__name__}")
        return True
    else:
        logger.warning("DSPy global LM configuration skipped: No Language Model could be initialised from AppConfig.")
        return False

if __name__ == '__main__':
    # Example of how to use this configuration module
    # For this example, we'll set up a basic console logger for the __main__ block
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    logger.info("Attempting to configure DSPy globally (example run)...")
    if configure_dspy_globally():
        logger.info("DSPy configured successfully (example run).")
        if dspy.settings.lm:
            logger.info(f"Current DSPy LM (example run): {dspy.settings.lm}")
        else:
            logger.warning("DSPy LM is not set after configuration attempt (example run).")
    else:
        logger.warning("DSPy configuration failed or was skipped (example run).")

    # You can also access config values directly:
    # logger.info(f"OpenAI Model configured (example run): {app_config.openai_model_name}")
    # if not app_config.openai_api_key:
    #     logger.warning("OpenAI API Key is not set in the environment (example run).")
