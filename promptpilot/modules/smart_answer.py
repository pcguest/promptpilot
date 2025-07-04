import dspy
# BasicQASignature is imported from the local signatures module.
from ..signatures import BasicQASignature

class SmartAnswerModule(dspy.Module):
    """
    A DSPy module for generating intelligent answers to questions using a defined signature.
    """
    def __init__(self):
        super().__init__()
        # The check for dspy.settings.lm is removed because app.py and test fixtures
        # now ensure an LM (real or dummy) is always configured before this module is instantiated.

        # Initialise the predictor with the imported signature.
        self.generate_answer = dspy.Predict(BasicQASignature)

    def forward(self, question: str) -> dspy.Prediction:
        """
        Generates an answer to the given question.

        Args:
            question: The question to be answered.

        Returns:
            A dspy.Prediction object containing the generated answer.
        """
        prediction = self.generate_answer(question=question)
        return prediction

if __name__ == '__main__':
    # This block demonstrates how to use the SmartAnswerModule independently.
    # It requires a Language Model (LM) to be configured first.
    #
    # To run this example:
    # 1. Ensure you have an LM provider configured (e.g., OpenAI API key in .env).
    # 2. Uncomment the following lines.
    # 3. Execute this script directly: `python promptpilot/modules/smart_answer.py`

    # print("--- SmartAnswerModule Standalone Example ---")
    # # Attempt to configure DSPy globally (example assumes config.py is in the parent directory)
    # import sys
    # import os
    # # Add project root to sys.path to allow `from promptpilot.config import ...`
    # project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # if project_root not in sys.path:
    #     sys.path.insert(0, project_root)
    #
    # from promptpilot.config import configure_dspy_globally
    #
    # if configure_dspy_globally():
    #     print("DSPy configured successfully for the example.")
    #     try:
    #         smart_answer_module = SmartAnswerModule()
    #         test_question = "What is the main purpose of the DSPy framework?"
    #         print(f"\nQuestion: \"{test_question}\"")
    #
    #         response = smart_answer_module(question=test_question)
    #         print(f"Answer: {response.answer}")
    #
    #         # You can inspect the history of interactions with the LM if supported
    #         # print("\n--- LM Interaction History (last 1) ---")
    #         # dspy.settings.inspect_history(n=1)
    #
    #     except dspy.DSPyError as e:
    #         print(f"\nA DSPyError occurred during the example: {e}")
    #     except Exception as e:
    #         print(f"\nAn unexpected error occurred during the example: {e}")
    # else:
    #     print("DSPy LM could not be configured. Skipping SmartAnswerModule example.")
    #     print("Please ensure your .env file and LM provider (e.g., OpenAI) are set up correctly.")
    #
    # print("\n--- End of SmartAnswerModule Standalone Example ---")
    pass # Keep the pass statement if the __main__ block is commented out.

    print("SmartAnswerModule defined. To run the standalone example, uncomment the code within the "
          "`if __name__ == '__main__':` block in this file and ensure your LM is configured (e.g., via .env).")
