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
    def __init__(self, model_name="dummy-model"): # Allow model_name override if needed
        super().__init__(model_name)
        self.provider = "dummy"
        self.history = [] # To inspect calls
        self.is_chat_model = True # Helps DSPy adapters determine how to format input

    def _get_raw_responses(self, prompt_text: str, **kwargs) -> list[str]:
        """
        Helper to determine response based on prompt content for testing.
        For DSPy 3.x with JSONAdapter, this should return a list of JSON strings.
        Each JSON string should represent a valid output structure for BasicQASignature.
        """
        if "EMPTY_STRING_RESPONSE" in prompt_text:
            # Valid JSON for an empty answer
            return ['{"answer": ""}']
        elif "EMPTY_LIST_RESPONSE" in prompt_text:
            # LM provides no valid completions (JSON strings)
            return []
        elif "NONE_IN_LIST_RESPONSE" in prompt_text:
            # This would be like malformed JSON, or non-JSON string.
            # JSONAdapter will fail to parse this.
            return ["<NoneResponseMarker>"] # This will cause a JSON parse error
        elif "MALFORMED_JSON_RESPONSE" in prompt_text:
            return ['{"answer": "incomplete json...'] # Malformed JSON
        elif "RAISE_ERROR_REQUEST" in prompt_text:
            raise ValueError("Simulated LLM Error during response generation")
        elif "What is DSPy?" in prompt_text:
            return ['{"answer": "DSPy is a programming model for prompting and composing LMs."}']
        elif "capital of France" in prompt_text:
            return ['{"answer": "Paris."}']
        else: # Default dummy answer
            return ['{"answer": "This is a dummy answer."}']

    def __call__(self, messages: list[dict[str,any]] | str , only_completed=True, return_sorted=False, **kwargs) -> list[str]:
        prompt_text_for_logic = ""
        actual_input_for_history = messages # Log what was actually received by __call__

        if isinstance(messages, str):
            prompt_text_for_logic = messages
        elif isinstance(messages, list) and messages:
            # Basic extraction: concatenate content from user/system messages for simplicity
            # or just use the last user/system message.
            # For this dummy LM, we'll assume the relevant trigger phrase is in the last content.
            prompt_text_for_logic = messages[-1].get('content', '')
        else:
            prompt_text_for_logic = str(messages)


        if "RAISE_ERROR_CALL" in prompt_text_for_logic:
            self.history.append({'type': '__call__', 'prompt_or_messages': actual_input_for_history, 'parsed_prompt_text': prompt_text_for_logic, 'kwargs': kwargs, 'error': 'Simulated LLM Error in __call__'})
            raise ConnectionError("Simulated LLM Connection Error in __call__")

        try:
            raw_responses = self._get_raw_responses(prompt_text_for_logic, **kwargs)
            self.history.append({
                'type': '__call__',
                'prompt_or_messages': actual_input_for_history,
                'parsed_prompt_text': prompt_text_for_logic,
                'kwargs': kwargs,
                'response': raw_responses
            })
            return raw_responses
        except Exception as e:
            self.history.append({'type': '__call__', 'prompt_or_messages': actual_input_for_history, 'parsed_prompt_text': prompt_text_for_logic, 'kwargs': kwargs, 'error': str(e)})
            raise


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
    """Test if SmartAnswerModule initialises correctly when an LM is configured."""
    try:
        module = SmartAnswerModule()
        assert module is not None
        assert hasattr(module, "generate_answer")
        # dspy.Predict stores the signature *class* if a class is passed to its constructor
        assert module.generate_answer.signature is BasicQASignature
    except RuntimeError as e: # Changed from dspy.DSPyError
        pytest.fail(f"SmartAnswerModule initialisation failed with configured LM: {e}")

def test_smart_answer_module_initialization_no_lm():
    """Test if SmartAnswerModule raises error if no LM is configured."""
    original_lm = dspy.settings.lm
    dspy.settings.configure(lm=None) # Explicitly set to None
    with pytest.raises(RuntimeError, match="DSPy LM not configured"): # Changed from dspy.DSPyError
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
    assert question in last_call['parsed_prompt_text'] # Check for substring


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
    """Test initialisation by mocking dspy.settings directly."""
    # Setup the mock for dspy.settings.lm
    mock_settings.lm = DummyLM()

    module = SmartAnswerModule()
    assert module is not None
    assert module.generate_answer.signature is BasicQASignature

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

    MockPredict.assert_called_once_with(BasicQASignature) # Check if dspy.Predict was initialised correctly
    mock_predictor_instance.assert_called_once_with(question=question) # Check if the predictor instance was called
    assert response.answer == "Mocked answer"

# Note: The `configured_dspy_for_test` fixture is generally a cleaner way to handle
# DSPy's global settings for integration-style tests of modules.
# Direct mocking like in `test_smart_answer_module_forward_with_mocked_predictor`
# can be useful for unit testing the module's own logic in isolation from dspy.Predict's behaviour.
# Choose the approach that best fits the testing goal.
# For this project, testing with a DummyLM via the fixture is a good balance.


def test_smart_answer_module_empty_question(configured_dspy_for_test):
    """Test SmartAnswerModule with an empty question string."""
    dummy_lm = configured_dspy_for_test
    module = SmartAnswerModule()
    response = module(question="")
    assert isinstance(response, dspy.Prediction)
    # Current DummyLM returns "This is a dummy answer." for unknown prompts
    assert response.answer == "This is a dummy answer."
    # The prompt generated by Predict for an empty question will still have the surrounding template.
    # The 'parsed_prompt_text' will be the full JSON-instruction template.
    # We check that the specific part for the question is empty within that template.
    history_prompt = dummy_lm.history[-1]['parsed_prompt_text']
    assert "[[ ## question ## ]]\n\n\nRespond with a JSON object" in history_prompt or \
           "[[ ## question ## ]]\n\nRespond with a JSON object" in history_prompt # Allowing for slight variations


def test_smart_answer_module_whitespace_question(configured_dspy_for_test):
    """Test SmartAnswerModule with a whitespace-only question string."""
    dummy_lm = configured_dspy_for_test
    module = SmartAnswerModule()
    question = "   \t   "
    response = module(question=question)
    assert isinstance(response, dspy.Prediction)
    assert response.answer == "This is a dummy answer." # As per DummyLM's default for non-specific content
    # Check that the whitespace question was passed to the LM's core logic by being part of the full prompt.
    assert question in dummy_lm.history[-1]['parsed_prompt_text']


def test_smart_answer_module_unicode_question(configured_dspy_for_test):
    """Test SmartAnswerModule with a Unicode question."""
    dummy_lm = configured_dspy_for_test
    module = SmartAnswerModule()
    question = "מה בירת צרפת?"  # "What is the capital of France?" in Hebrew
    response = module(question=question)
    assert isinstance(response, dspy.Prediction)
    assert response.answer == "This is a dummy answer."
    assert question in dummy_lm.history[-1]['parsed_prompt_text']


def test_smart_answer_module_lm_returns_empty_string(configured_dspy_for_test):
    """Test module behaviour when LM returns an empty string."""
    dummy_lm = configured_dspy_for_test
    module = SmartAnswerModule()
    question_trigger = "Activate EMPTY_STRING_RESPONSE mode"
    response = module(question=question_trigger)
    assert isinstance(response, dspy.Prediction)
    assert response.answer == ""
    assert "EMPTY_STRING_RESPONSE" in dummy_lm.history[-1]['parsed_prompt_text']


def test_smart_answer_module_lm_returns_empty_list(configured_dspy_for_test):
    """Test module behaviour when LM returns an empty list (no completions)."""
    dummy_lm = configured_dspy_for_test
    module = SmartAnswerModule()
    question_trigger = "Activate EMPTY_LIST_RESPONSE mode"

    # Based on DSPy 3.0.0b2 dspy/predict/predict.py L200 `self.signature.ensure_valid(parsed_completions[0], ...) `
    # If parsed_completions is empty (because LM returned []), this will cause an IndexError.
    # Update: With JSONAdapter, if the LM returns [], Predict might set field to None or empty.
    response = module(question=question_trigger)
    assert isinstance(response, dspy.Prediction)
    # The actual behavior when LM returns [] needs to be confirmed.
    # If Predict can't get a completion, it might leave the field as None or an empty string.
    # Given JSON parsing, if no JSON is found (empty list from LM), answer would likely be None.
    # When no fields are populated, the Prediction object might not have the attribute or its value is None.
    # A robust check is that the internal store is empty or the specific key is absent/None.
    assert len(response.keys()) == 0 or getattr(response, 'answer', None) is None

    assert "EMPTY_LIST_RESPONSE" in dummy_lm.history[-1]['parsed_prompt_text']
    # This history entry is recorded in DummyLM before the error in Predict, which is fine.


def test_smart_answer_module_lm_call_raises_error(configured_dspy_for_test):
    """Test module behaviour when the LM __call__ itself raises an error (e.g., network)."""
    dummy_lm = configured_dspy_for_test
    module = SmartAnswerModule()
    question_trigger = "Activate RAISE_ERROR_CALL mode"

    with pytest.raises(ConnectionError, match="Simulated LLM Connection Error in __call__"):
        module(question=question_trigger)

    assert "RAISE_ERROR_CALL" in dummy_lm.history[-1]['parsed_prompt_text']
    assert dummy_lm.history[-1]['error'] == 'Simulated LLM Error in __call__'


def test_smart_answer_module_lm_response_gen_raises_error(configured_dspy_for_test):
    """Test module behaviour when LM raises an error during response generation phase."""
    dummy_lm = configured_dspy_for_test
    module = SmartAnswerModule()
    question_trigger = "Activate RAISE_ERROR_REQUEST mode"

    with pytest.raises(ValueError, match="Simulated LLM Error during response generation"):
        module(question=question_trigger)

    assert "RAISE_ERROR_REQUEST" in dummy_lm.history[-1]['parsed_prompt_text']
    assert "Simulated LLM Error during response generation" in dummy_lm.history[-1]['error']
