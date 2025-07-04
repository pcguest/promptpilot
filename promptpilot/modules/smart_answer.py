import dspy
# Updated import: BasicQASignature is now in promptpilot.signatures
from ..signatures import BasicQASignature

class SmartAnswerModule(dspy.Module):
    """
    A DSPy module for generating smart answers to questions using a defined signature.
    """
    def __init__(self):
        super().__init__()
        # Ensure that an LM is configured globally before initializing predictors.
        # dspy.Predict requires dspy.settings.lm to be available at initialization.
        if not dspy.settings.lm:
            # This error message guides the user to configure DSPy globally.
            # The actual configuration should happen at application startup (e.g., in app.py or config.py).
            raise dspy.DSPyError(
                "DSPy LM not configured. SmartAnswerModule requires a globally configured LLM. "
                "Please call `configure_dspy_globally()` or `dspy.settings.configure(lm=your_lm)` "
                "before instantiating this module."
            )

        # Initialize the predictor with the imported signature
        self.generate_answer = dspy.Predict(BasicQASignature)

    def forward(self, question: str) -> dspy.Prediction:
        """
        Generates an answer to the given question.

        Args:
            question: The question to answer.

        Returns:
            A dspy.Prediction object containing the answer.
        """
        prediction = self.generate_answer(question=question)
        return prediction

if __name__ == '__main__':
    # This example demonstrates how to use the SmartAnswerModule.
    # It requires an LLM to be configured.
    # For example, using OpenAI:
    #
    # import os
    # from dspy.openai import OpenAI
    #
    # # Ensure your OPENAI_API_KEY environment variable is set, or pass api_key directly.
    # if not os.getenv("OPENAI_API_KEY"):
    #     print("Error: OPENAI_API_KEY environment variable not set.")
    #     print("Please set it or configure the LLM with an API key directly.")
    #     # Example: llm = OpenAI(model="gpt-3.5-turbo", api_key="sk-...")
    # else:
    #     llm = OpenAI(model="gpt-3.5-turbo")
    #     dspy.settings.configure(lm=llm)

    #     try:
    #         if dspy.settings.lm:
    #             smart_answer_module = SmartAnswerModule()
    #             test_question = "What is the main purpose of the DSPy framework?"
    #             response = smart_answer_module(question=test_question)
    #             print(f"Question: {test_question}")
    #             print(f"Answer: {response.answer}")
    #
    #             # You can inspect the history of interactions with the LM
    #             # dspy.settings.inspect_history(n=1)
    #         else:
    #             print("DSPy LM not configured. Skipping SmartAnswerModule example usage.")
    #             print("To run this example, configure an LM like OpenAI: ")
    #             print("  from dspy.openai import OpenAI")
    #             print("  llm = OpenAI(model=\"gpt-3.5-turbo\", api_key=\"YOUR_API_KEY\")")
    #             print("  dspy.settings.configure(lm=llm)")
    #
    #     except Exception as e:
    #         print(f"Error in SmartAnswerModule example: {e}")
    #         print("This might be due to missing API keys or incorrect LLM configuration.")

    print("SmartAnswerModule defined. To run the example, uncomment the __main__ block and configure an LLM.")
    pass
