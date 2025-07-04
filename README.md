# PromptPilot

**Orchestrate LLM prompts in a clean, modular, and testable way using DSPy.**

PromptPilot aims to provide a clear, minimal, and human-centric approach to building applications with Large Language Models. It emphasises readability, modularity, good development practices, and leverages the DSPy framework for effective prompt engineering and optimisation.

This project is designed to be a learning tool and a template for building more complex LLM applications.

## Core Ideas
- **Modular Design**: Structure your LLM logic into reusable DSPy modules (see `promptpilot/modules/`).
- **Testability**: Facilitate testing of individual prompt components and overall application flow (see `tests/`).
- **Clarity**: Keep the codebase simple, understandable, and easy to learn from.
- **Modern DSPy Integration**: Utilise DSPy v3+ features for building and configuring LLM pipelines.
- **Configuration Management**: Easy setup for LLM providers using environment variables.

## Project Structure
```
.
├── .env.example         # Example environment variables
├── .gitignore
├── LICENSE
├── README.md
├── promptpilot
│   ├── __init__.py
│   ├── app.py           # Main application entry point & example usage
│   ├── config.py        # Configuration loading (incl. LLM setup)
│   ├── modules
│   │   ├── __init__.py
│   │   └── smart_answer.py  # Example DSPy module
│   └── signatures.py    # DSPy Signatures
├── requirements.txt     # Project dependencies
├── ruff.toml            # Linter/formatter configuration for Ruff
├── streamlit_app.py     # Optional Streamlit UI for demo
└── tests
    ├── __init__.py
    └── test_smart_answer.py # Tests for SmartAnswerModule
```
(Note: `config.py` and `signatures.py` were missing from the original README's structure, added for completeness.)

## Getting Started

### Prerequisites
- Python 3.8+
- An API key for an LLM provider (e.g., OpenAI).

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/promptpilot.git # Replace with actual repo URL
cd promptpilot
```

### 2. Create and Activate a Virtual Environment
It's highly recommended to use a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```
This will install `dspy-ai`, `python-dotenv`, and other necessary packages. If you want to use a specific LLM provider like OpenAI, you might need to install their library separately if not already included or uncomment it in `requirements.txt` and re-run pip install:
```bash
pip install openai # Example for OpenAI
```

### 4. Configure Environment Variables
The application uses a `.env` file to manage API keys and other configurations.

A `.env.example` file is provided in the root of the project. Copy this file to `.env` and then add your actual LLM API key and any other configurations:
```bash
cp .env.example .env
```
Then, edit `.env`. For example, for OpenAI:
```env
OPENAI_API_KEY="your_openai_api_key_here"
# You can also specify a model name, otherwise it defaults to gpt-3.5-turbo
# OPENAI_MODEL_NAME="gpt-4"
```
The `promptpilot/config.py` script (used by `app.py`) will load these variables.

### 5. Run the Application
To run the example application which uses the `SmartAnswerModule`:
```bash
python promptpilot/app.py
```
You should see output indicating the DSPy setup and then a question being asked to the `SmartAnswerModule`, followed by the LLM's answer.

### 6. Run Tests
To ensure everything is set up correctly and the modules are working as expected, first ensure `pytest` is installed (it's included in `requirements.txt`):
```bash
pip install -r requirements.txt # Ensures pytest and other dependencies are installed
python -m pytest
```
The tests use a `DummyLM` and do not require an active API key to run, as they mock the LLM responses.

## Usage

### `SmartAnswerModule`
Located in `promptpilot/modules/smart_answer.py`, this module is a simple example of a DSPy `Module` that uses a `Predict`or with a defined `Signature` (`BasicQASignature`) to answer questions.

You can adapt this module or create new ones in the `promptpilot/modules/` directory following similar patterns.

### Configuring DSPy
DSPy is configured in `promptpilot/app.py` within the `configure_dspy_settings()` function. Currently, it's set up for OpenAI but can be adapted for other LLM providers supported by DSPy (e.g., Cohere, Anthropic, HuggingFace models). You would typically need to:
1. Install the required Python library for that provider (e.g., `pip install anthropic`).
2. Modify `configure_dspy_settings()` to import and instantiate the correct DSPy LM client (e.g., `dspy.Anthropic()`).
3. Update your `.env` file with the necessary API keys or environment variables for that provider.

## Development

### Logging
The application uses Python's built-in `logging` module.
- Basic configuration is set up in `promptpilot/app.py`.
- Messages are logged to the console with timestamps and log levels.
- You can adjust the logging level (e.g., to `DEBUG`) in `app.py` if needed.

### Linting and Formatting
This project now includes a basic configuration for [Ruff](https://beta.ruff.rs/docs/) in `ruff.toml`. Ruff is an extremely fast Python linter and formatter, written in Rust.

**To use Ruff:**
1.  Install Ruff (if not already included in your global tools or project's dev dependencies):
    ```bash
    pip install ruff
    ```
2.  Check for linting issues:
    ```bash
    ruff check .
    ```
3.  Apply auto-fixes (for many issues):
    ```bash
    ruff check . --fix
    ```
4.  Format code using Ruff's formatter (as an alternative to Black):
    ```bash
    ruff format .
    ```
The `ruff.toml` file is pre-configured with some common rule sets (like Pyflakes, pycodestyle) and a line length of 88. You can customise this file further based on your preferences.

### Dependency Management
Currently, dependencies are managed via `requirements.txt`. For more advanced dependency management, consider using tools like Poetry or PDM, which use `pyproject.toml`. This can provide better separation of dependencies, lock files, and packaging features.

## Optional Streamlit UI
An optional Streamlit application is provided in `streamlit_app.py` for a simple web-based UI to interact with the `SmartAnswerModule`.

**To run the Streamlit app:**
1.  Ensure Streamlit is installed:
    ```bash
    pip install streamlit
    ```
    (Consider adding `streamlit` to your `requirements.txt` if you use this regularly).
2.  Make sure your `.env` file is correctly configured with your LLM API keys.
3.  From the project root directory, run:
    ```bash
    streamlit run streamlit_app.py
    ```
This will typically open the application in your web browser.

## Contributing
Contributions are welcome! If you have suggestions for improvements, new features, or find any bugs, please open an issue or submit a pull request.

## Licence
This project is licenced under the MIT Licence - see the [LICENSE](LICENSE) file for details.

---
*This project structure and initial setup were developed with the assistance of an AI coding agent to align with modern Python and DSPy best practices.*
