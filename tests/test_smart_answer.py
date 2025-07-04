import pytest
import dspy
from unittest.mock import patch, MagicMock

# Since SmartAnswerModule is in the parent directory's `promptpilot` package,
# we need to adjust the Python path if running pytest from the root or `tests` directory.
# This is a common pattern. Alternatively, install the package in editable mode (pip install -e .).
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from promptpilot.modules.smart_answer import SmartAnswerModule, BasicQASignature

# Dummy LM for testing purposes
class DummyLM(dspy.LM):
    def __init__(self):
        super().__init__("dummy-model")
        self.provider = "dummy"
        self.history = [] # To inspect calls

    def basic_request(self, prompt, **kwargs):
        # Simulate a response structure similar to what a real LM would return
        # This needs to match what dspy.Predict expects after parsing.
        # For a simple signature like "question -> answer", the LM is expected
        # to return a string that Predict will parse.
        # If the prompt contains "What is DSPy?", return a fixed answer.
        self.history.append({'prompt': prompt, 'kwargs': kwargs, 'response': "DSPy is a framework."})
        return ["DSPy is a framework."] # dspy.Predict expects a list of choices

    def __call__(self, prompt, only_completed=True, return_sorted=False, **kwargs):
        # Simulate the behavior of dspy.Predict calling the LM
        # It expects a list of completions (strings).
        if "What is DSPy?" in prompt:
            response = "DSPy is a programming model for prompting and composing LMs."
        elif "capital of France" in prompt:
            response = "Paris."
        else:
            response = "This is a dummy answer."

        # Store the interaction
        self.history.append({
            'prompt': prompt,
            'kwargs': kwargs,
            'response': [response] # dspy.Predict expects a list of choices
        })
        return [response] # Return a list of choices

@pytest.fixture(scope="module")
def configured_dspy_for_test():
    """Fixture to configure DSPy with a dummy LM for tests."""
    if not dspy.settings.lm: # Configure only if not already configured
        original_lm = dspy.settings.lm
        dummy_lm = DummyLM()
        dspy.settings.configure(lm=dummy_lm)
        yield dummy_lm # Provide the dummy_lm to the test if needed
        dspy.settings.configure(lm=original_lm) # Restore original settings
    else:
        # If an LM is already configured (e.g. globally for other tests or by user)
        # We can choose to skip, or use it, or override. For now, let's use the dummy one.
        original_lm = dspy.settings.lm
        dummy_lm = DummyLM()
        dspy.settings.configure(lm=dummy_lm)
        yield dummy_lm
        dspy.settings.configure(lm=original_lm)


def test_smart_answer_module_initialization(configured_dspy_for_test):
    """Test if SmartAnswerModule initializes correctly when an LM is configured."""
    try:
        module = SmartAnswerModule()
        assert module is not None
        assert hasattr(module, "generate_answer")
        assert isinstance(module.generate_answer.signature, BasicQASignature)
    except dspy.DSPyError as e:
        pytest.fail(f"SmartAnswerModule initialization failed with configured LM: {e}")

def test_smart_answer_module_initialization_no_lm():
    """Test if SmartAnswerModule raises error if no LM is configured."""
    original_lm = dspy.settings.lm
    dspy.settings.configure(lm=None) # Explicitly set to None
    with pytest.raises(dspy.DSPyError, match="DSPy LM not configured"):
        SmartAnswerModule()
    dspy.settings.configure(lm=original_lm) # Restore

def test_smart_answer_module_forward_pass(configured_dspy_for_test):
    """Test the forward pass of SmartAnswerModule with the dummy LM."""
    dummy_lm = configured_dspy_for_test
    module = SmartAnswerModule()
    question = "What is DSPy?"

    # The dummy LM's __call__ should be invoked by dspy.Predict
    response = module(question=question)

    assert isinstance(response, dspy.Prediction)
    assert "answer" in response
    assert response.answer == "DSPy is a programming model for prompting and composing LMs."

    # Check if the dummy LM was called as expected
    assert len(dummy_lm.history) > 0
    # The prompt sent to the LM by dspy.Predict will be structured based on the signature
    # For BasicQASignature: "Answers questions.\n\n---\n\nQuestion: What is DSPy?\nAnswer:" (or similar)
    # We can make this test more robust by inspecting the prompt more closely if needed.
    last_call = dummy_lm.history[-1]
    assert "Question: What is DSPy?" in last_call['prompt']


# To run these tests, navigate to the project root directory and run:
# python -m pytest
#
# Make sure pytest is installed: pip install pytest
# And that your project structure allows importing promptpilot:
# One way is to `pip install -e .` from the root directory (if you add a setup.py)
# Or adjust PYTHONPATH, or use the sys.path hack as shown above.
#
# The sys.path hack is used here for simplicity without requiring project installation.
# For larger projects, `pip install -e .` is recommended.

# Example of how to mock dspy.settings.lm if direct patching is preferred for some tests:
@patch('dspy.settings')
def test_smart_answer_module_init_with_mocked_settings(mock_settings):
    """Test initialization by mocking dspy.settings directly."""
    # Setup the mock for dspy.settings.lm
    mock_settings.lm = DummyLM()

    module = SmartAnswerModule()
    assert module is not None
    assert isinstance(module.generate_answer.signature, BasicQASignature)

@patch('dspy.Predict')
def test_smart_answer_module_forward_with_mocked_predictor(MockPredict, configured_dspy_for_test):
    """Test forward pass by mocking dspy.Predict to isolate module logic."""
    # Instance of the mocked dspy.Predict
    mock_predictor_instance = MagicMock()
    mock_predictor_instance.return_value = dspy.Prediction(answer="Mocked answer")

    # Configure MockPredict to return our instance when called (e.g. dspy.Predict(BasicQASignature))
    MockPredict.return_value = mock_predictor_instance

    module = SmartAnswerModule() # This will now use the mocked dspy.Predict

    question = "Any question"
    response = module(question=question)

    MockPredict.assert_called_once_with(BasicQASignature) # Check if dspy.Predict was initialized correctly
    mock_predictor_instance.assert_called_once_with(question=question) # Check if the predictor instance was called
    assert response.answer == "Mocked answer"

# Note: The `configured_dspy_for_test` fixture is generally a cleaner way to handle
# DSPy's global settings for integration-style tests of modules.
# Direct mocking like in `test_smart_answer_module_forward_with_mocked_predictor`
# can be useful for unit testing the module's own logic in isolation from dspy.Predict's behavior.
# Choose the approach that best fits the testing goal.
# For this project, testing with a DummyLM via the fixture is a good balance.
