import pytest
import dspy

# Ensure the package root is in sys.path for direct script execution from tests directory
import sys
import os
if __package__ is None or __package__ == '':
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)

from promptpilot.modules.smart_answer import SmartAnswerModule

# Dummy language model that mimics an LLM. Must inherit from dspy.LM.
class DummyLM(dspy.LM):
    def __init__(self, model_name="dummy-test-model"): # LMs usually take a model name
        super().__init__(model_name) # Pass model_name to super
        self.provider = "dummy" # Standard LM attribute
        self.was_called = False # To check if the LM was invoked

    def __call__(self, *args, **kwargs): # __call__ is fine for a dspy.Module based LM
        self.was_called = True
        # User suggestion: return dspy.Prediction(answer="Dummy answer")
        # For this to work directly, SmartAnswerModule's Predictor would need to be
        # configured to expect a Prediction object from the LM, which is not standard.
        # Standard LMs provide string/dict completions that Predict then turns into a Prediction.
        # To align with how dspy.Predict and its adapters (e.g., JSONAdapter for structured output)
        # typically work with signatures like BasicQASignature (which expects an 'answer' field),
        # the LM should output a raw completion (e.g., a JSON string).
        import json
        response_text = "Dummy answer"
        # BasicQASignature implies an "answer" field. JSONAdapter expects a JSON string.
        return [json.dumps({"answer": response_text})]


@pytest.fixture(autouse=True)
def configure_dspy_for_test():
    """
    Automatically configures DSPy with the DummyLM for all tests in this module.
    Uses 'lm' for dspy-ai 3.x instead of 'default_lm'.
    """
    dspy.settings.configure(lm=DummyLM())

def test_smart_answer_module_instantiation():
    """
    Tests if SmartAnswerModule can be instantiated when an LM is auto-configured.
    This implicitly tests that SmartAnswerModule doesn't error out if its internal
    LM check is removed (as an LM is now always configured by the fixture).
    """
    try:
        module = SmartAnswerModule()
        assert module is not None, "SmartAnswerModule should be successfully instantiated."
    except Exception as e:
        pytest.fail(f"SmartAnswerModule instantiation failed: {e}")

def test_forward_works():
    """
    Tests the forward pass of SmartAnswerModule using the auto-configured DummyLM,
    based on the user's suggested test structure.
    """
    module = SmartAnswerModule()

    # Retrieve the LM instance from dspy.settings to check its state
    # This is valid as the autouse fixture has already configured it.
    lm_instance = dspy.settings.lm
    assert isinstance(lm_instance, DummyLM), "The configured LM should be our DummyLM instance."

    question = "Who is the Prime Minister of the UK?"
    # Using module.forward() as suggested, which is equivalent to module() for dspy.Module
    result = module.forward(question=question)

    assert hasattr(result, "answer"), "The result should have an 'answer' attribute."
    # The DummyLM (as modified by me) returns a JSON string {"answer": "Dummy answer"}
    # which dspy.Predict parses into result.answer = "Dummy answer"
    assert "Dummy answer" in result.answer, "The answer should be from the DummyLM."
    assert lm_instance.was_called, "The DummyLM's __call__ method should have been invoked."

# Note: The original user suggestion had DummyLM return `dspy.Prediction(answer="Dummy answer")`.
# This has been adjusted in this DummyLM to `return [json.dumps({"answer": "Dummy answer"})]`
# because dspy.Predict expects LMs to provide raw string/dict completions, not Prediction objects.
# The Predict module itself is responsible for creating the Prediction object from these raw completions.
# This change makes the DummyLM behave more like a standard DSPy LM integration.
