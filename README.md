# PromptPilot

**Orchestrate LLM prompts in a clean, modular, and testable manner using DSPy.**

PromptPilot aims to provide a clear, minimal, and human-centric approach to building applications with Large Language Models (LLMs). It emphasises readability, modularity, good development practices, and leverages the DSPy framework for effective prompt engineering and optimisation.

This project is designed as a learning tool and a template for constructing more complex LLM applications.

## Core Concepts
- **Modular Design**: Structure your LLM logic into reusable DSPy modules (see `promptpilot/modules/`).
- **Testability**: Facilitate the testing of individual prompt components and overall application flow (see `tests/`).
- **Clarity**: Maintain a codebase that is simple, understandable, and easy to learn from.
- **Modern DSPy Integration**: Utilise DSPy v3+ features for building and configuring LLM pipelines.
- **Configuration Management**: Straightforward setup for LLM providers using environment variables.

## Project Structure
```
.
├── .gitignore
├── LICENSE
├── README.md
├── promptpilot
│   ├── __init__.py
│   ├── app.py           # Main application entry point & example usage
│   ├── config.py        # Configuration loading (e.g., API keys)
│   ├── modules
│   │   ├── __init__.py
│   │   └── smart_answer.py  # Example DSPy module
│   └── signatures.py    # DSPy signature definitions
├── requirements.txt     # Project dependencies
└── tests
    ├── __init__.py
    └── test_smart_answer.py # Tests for SmartAnswerModule
```
*(Note: The project structure displayed here includes `config.py` and `signatures.py` for completeness, reflecting the actual codebase.)*

## Getting Started

### Prerequisites
- Python 3.8 or newer.
- An API key for an LLM provider (e.g., OpenAI).

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/promptpilot.git # Please replace with the actual repository URL
cd promptpilot
```

### 2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage project dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
Install the core dependencies:
```bash
pip install -r requirements.txt
```
This will install `dspy-ai`, `python-dotenv`, and other essential packages.

If you intend to use a specific LLM provider, such as OpenAI, you may need to install its Python library. Uncomment the relevant line in `requirements.txt` and run `pip install -r requirements.txt` again, or install it directly:
```bash
pip install openai # Example for installing the OpenAI library
```
Refer to `requirements.txt` for a list of common provider libraries.

### 4. Configure Environment Variables
The application employs a `.env` file to manage API keys and other configurations.

Create a file named `.env` in the root of the project directory. Add your LLM API key to this file. For example, for OpenAI:
```env
OPENAI_API_KEY="your_openai_api_key_here"
# You can also specify a model name; otherwise, it defaults to a model like "gpt-3.5-turbo".
# OPENAI_MODEL_NAME="gpt-4"
```
The `promptpilot/config.py` module, utilised by `promptpilot/app.py`, will load these variables.

### 5. Run the Application
To execute the example application, which uses the `SmartAnswerModule`:
```bash
python promptpilot/app.py
```
You should observe output indicating the DSPy setup, followed by a question being posed to the `SmartAnswerModule` and the LLM's response.

### 6. Run Tests
To verify that everything is configured correctly and the modules are functioning as expected:
```bash
# Ensure pytest is installed (it's listed in requirements.txt as a development tool)
# If you haven't installed development tools: pip install pytest
python -m pytest
```
The tests primarily use a `DummyLM` (a mock Language Model) and generally do not require an active API key to run, as they simulate LLM responses for testing purposes.

## Usage

### `SmartAnswerModule`
Located in `promptpilot/modules/smart_answer.py`, this module serves as a simple example of a DSPy `Module`. It uses a `dspy.Predict`or in conjunction with a defined `Signature` (`BasicQASignature` from `promptpilot/signatures.py`) to answer questions.

You can adapt this module or create new ones within the `promptpilot/modules/` directory, following similar design patterns.

### Configuring DSPy
DSPy's global Language Model is configured in `promptpilot/config.py` via the `configure_dspy_globally()` function, which is called from `promptpilot/app.py`.
The current setup in `config.py` primarily demonstrates configuration for OpenAI but can be extended for other LLM providers supported by DSPy (e.g., Cohere, Anthropic, HuggingFace models). To use a different provider, you would typically need to:

1.  **Install the provider's library**: For instance, `pip install anthropic` for Anthropic models. Ensure the relevant library is listed and uncommented in `requirements.txt` or installed manually.
2.  **Modify `promptpilot/config.py`**:
    *   Update the `AppConfig` class to load necessary API keys and settings for the new provider from environment variables (loaded from `.env`).
    *   Adjust the `get_configured_lm()` method within `AppConfig` to instantiate and return the correct DSPy LM client for that provider (e.g., `dspy.Anthropic()`).
3.  **Update your `.env` file**: Add the required API keys and any other environment variables specific to the new provider.

## Development

### Linting and Formatting
While this project does not currently enforce a specific linter or formatter through automated checks, tools such as Ruff, Black, or Flake8 are recommended for maintaining code quality and consistency.
```bash
# Example using Ruff (install with: pip install ruff)
# ruff check .  # Check for linting issues
# ruff format . # Format code
```

### Dependency Management
Dependencies are currently managed via `requirements.txt`. For more advanced dependency management, especially in larger projects, consider using tools like Poetry or PDM. These tools often use `pyproject.toml` and can offer more robust dependency resolution, lock files, and packaging features.

## Contributing
Contributions are warmly welcomed! If you have suggestions for improvements, new features, or identify any bugs, please feel free to open an issue or submit a pull request.

## Licence
This project is licensed under the MIT Licence - see the [LICENSE](LICENSE) file for details.

---
*This project structure and initial setup were developed with the assistance of an AI coding agent to align with modern Python and DSPy best practices.*
