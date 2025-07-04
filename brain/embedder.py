import os
import openai
from dotenv import load_dotenv

class Embedder:
    """
    A wrapper class for generating text embeddings using the OpenAI API.

    This class handles loading the OpenAI API key from environment variables
    and provides a method to get embeddings for a given text string.
    It is configured to use the "text-embedding-ada-002" model.
    """

    def __init__(self):
        """
        Initialises the Embedder.

        Loads the OpenAI API key from the .env file or environment variables.
        Raises:
            ValueError: If the OpenAI API key is not found.
        """
        load_dotenv()
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key not found. Please set OPENAI_API_KEY in your .env file or environment variables."
            )
        openai.api_key = self.api_key
        self.model_name = "text-embedding-ada-002"

    def get_embedding(self, text: str) -> list[float]:
        """
        Generates an embedding for the given text.

        Args:
            text: The text string to embed.

        Returns:
            A list of floats representing the embedding.

        Raises:
            openai.APIError: If there is an issue with the OpenAI API call.
            Exception: For other unexpected errors during embedding generation.
        """
        if not text or not isinstance(text, str):
            raise ValueError("Input text must be a non-empty string.")

        try:
            text = text.replace("\n", " ") # OpenAI recommendation
            response = openai.embeddings.create(input=[text], model=self.model_name)
            return response.data[0].embedding
        except openai.APIError as e:
            # Log the error or handle it more gracefully
            print(f"OpenAI API error: {e}")
            raise
        except Exception as e:
            # Log unexpected errors
            print(f"An unexpected error occurred while generating embedding: {e}")
            raise

if __name__ == '__main__':
    # Example usage (requires a .env file with OPENAI_API_KEY)
    try:
        embedder = Embedder()
        sample_text = "This is a test sentence for the Embedder class."
        embedding = embedder.get_embedding(sample_text)
        print(f"Embedding for '{sample_text}':")
        print(embedding[:5]) # Print first 5 dimensions as an example
        print(f"Embedding dimension: {len(embedding)}")

        # Test empty string
        # embedder.get_embedding("") # Should raise ValueError

    except ValueError as ve:
        print(f"Configuration error: {ve}")
    except openai.APIError as apie:
        print(f"API error during example usage: {apie}")
    except Exception as ex:
        print(f"An unexpected error occurred during example usage: {ex}")
