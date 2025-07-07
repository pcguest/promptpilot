"""Command-line interface for the PromptPilot application.

This module provides a CLI for interacting with the PromptPilot agents. It is
built with click and is designed to be a user-friendly and extensible
interface.
"""

import click

from .components.smart_answer_agent import SmartAnswerAgent
from .config import configure_dspy


@click.group()
def main():
    """PromptPilot: A DSPy-powered CLI tool for LLM orchestration."""
    pass


@main.command()
@click.argument("question", nargs=-1)
@click.option("--debug", is_flag=True, help="Enable debug mode.")
def ask(question: tuple[str], debug: bool):
    """Ask a question to the SmartAnswerAgent."""
    question_text = " ".join(question)

    if not question_text:
        click.echo("Please provide a question.")
        return

    try:
        configure_dspy()
        agent = SmartAnswerAgent()
        answer = agent.forward(question=question_text)
        click.echo(f"💡 {answer}")

    except ValueError as e:
        click.echo(f"Error: {e}", err=True)
    except ImportError as e:
        click.echo(f"Error: {e}", err=True)
    except RuntimeError as e:
        click.echo(f"Error: {e}", err=True)
    except Exception as e:
        click.echo(f"An unexpected error occurred: {e}", err=True)


if __name__ == "__main__":
    main()
