"""Configuration for the PromptPilot application.

This module handles the loading of environment variables and configures the DSPy
settings. It is designed to be the central point for all configuration,
making it easier to manage settings across the application.
"""

import os

import dotenv
import dspy


def configure_dspy() -> None:
    """Configures the DSPy framework with the language model.

    This function loads the OpenAI API key from the .env file and sets up the
    language model for DSPy. It is designed to fail gracefully if the API
    key is not found, providing a clear error message.
    """
    dotenv.load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if not openai_api_key:
        raise ValueError("OPENAI_API_KEY not found in .env file.")

    try:
        from dspy.llms import ChatGPT

        llm = ChatGPT(model="gpt-4", api_key=openai_api_key)
        dspy.settings.configure(lm=llm)
    except ImportError:
        raise ImportError("DSPy is not installed. Please run 'pip install dspy-ai'.")
    except Exception as e:
        raise RuntimeError(f"Failed to configure the language model: {e}")
