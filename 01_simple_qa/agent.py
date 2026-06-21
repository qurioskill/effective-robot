"""Example 1 — The simplest possible agent: ask once, answer once, exit.

This is the "hello world" of LLM agents. There are no tools, no loops, and no
memory. We send one question to the model and print one answer.

Flow
----

    +-----------+      question      +-------------+      answer
    |   user    | -----------------> |   ChatLLM   | -----------> stdout
    +-----------+                    +-------------+

Run it::

    python 01_simple_qa/agent.py
    python 01_simple_qa/agent.py "What is a vector database?"
"""

from __future__ import annotations

import os
import sys

# --- Make the top-level ``shared`` package importable -----------------------
# When you run ``python 01_simple_qa/agent.py`` the script's own folder is on
# ``sys.path``, but the project root is not. Adding the project root lets us do
# ``from shared.llm import get_llm`` no matter where you launch from.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import HumanMessage, SystemMessage  # noqa: E402

from shared.llm import get_llm  # noqa: E402


# A system message sets the model's role/behaviour. The human message is the
# actual question. This two-message pattern is the foundation everything else
# builds on.
SYSTEM_PROMPT = "You are a concise, helpful assistant. Answer in a few sentences."


def ask(question: str) -> str:
    """Send a single question to the model and return its answer.

    Args:
        question: The user's question.

    Returns:
        The model's answer as plain text.
    """
    llm = get_llm()

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    # ``invoke`` makes one round-trip to the API and returns an AIMessage.
    response = llm.invoke(messages)
    return response.content


def main() -> None:
    """Entry point: read a question from the CLI args (or a default) and print."""
    # Allow passing the question as a command-line argument; otherwise use a
    # sensible default so the example runs with zero typing.
    if len(sys.argv) > 1:
        question = " ".join(sys.argv[1:])
    else:
        question = "In one sentence, what is an AI agent?"

    print(f"Q: {question}")
    answer = ask(question)
    print(f"A: {answer}")


if __name__ == "__main__":
    main()
