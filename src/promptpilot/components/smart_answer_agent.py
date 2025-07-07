"""The core agent for providing smart answers.

This module contains the SmartAnswerAgent, a DSPy module that uses a
predictive model to generate answers to user questions. It is designed to be
a self-contained component that can be easily integrated into different
interfaces, such as a CLI or a web application.
"""

import dspy

from promptpilot.signatures import SmartAnswerSignature


class SmartAnswerAgent(dspy.Module):
    """A DSPy module for providing smart answers to user questions."""

    def __init__(self):
        """Initialises the SmartAnswerAgent."""
        super().__init__()
        self.predictor = dspy.Predict(SmartAnswerSignature)

    def forward(self, question: str) -> str:
        """Generates an answer to the given question.

        Args:
            question: The user's question.

        Returns:
            The generated answer.
        """
        result = self.predictor(question=question)
        return result.answer
