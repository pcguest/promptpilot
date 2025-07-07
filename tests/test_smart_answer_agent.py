"""Unit tests for the SmartAnswerAgent.

This module contains unit tests for the SmartAnswerAgent, using a DummyLM to
simulate the behaviour of a language model. This allows for fast and reliable
testing of the agent's logic without making actual API calls.
"""

import dspy
import pytest

from promptpilot.components.smart_answer_agent import SmartAnswerAgent


class DummyLM(dspy.LM):
    """A dummy language model for testing."""

    def __init__(self):
        super().__init__("")

    def __call__(self, messages, **kwargs):
        return [{'text': '{"answer": "dummy response"}'}]


@pytest.fixture
def smart_answer_agent():
    """Fixture for the SmartAnswerAgent."""
    dspy.settings.configure(lm=DummyLM())
    return SmartAnswerAgent()


def test_smart_answer_agent(smart_answer_agent):
    """Test the SmartAnswerAgent."""
    question = "What is the capital of France?"
    answer = smart_answer_agent.forward(question)
    assert isinstance(answer, str)
    assert answer == "dummy response"
