"""Signatures for the PromptPilot application.

This module defines the DSPy signatures, which are the core of the
prompting strategy. By centralising the signatures, we can easily reuse them
across different components and maintain a clear separation of concerns.
"""

import dspy


class SmartAnswerSignature(dspy.Signature):
    """Answer clearly, concisely, and with evidence if needed."""

    question = dspy.InputField(desc="The user's question")
    answer = dspy.OutputField(desc="A helpful, fact-based, natural response")
