"""Example 2 — The same Q/A agent, but running continuously (a chat loop).

This adds two things to Example 1:

1. A **REPL loop** so the user can keep asking questions.
2. **Conversation history** so the model remembers earlier turns.

Flow
----

    +--------+   question   +-----------+   answer   +--------+
    |  user  | -----------> |  ChatLLM  | ---------> |  user  |
    +--------+              +-----------+            +--------+
        ^                        |                        |
        |     history grows      |   append both          |
        +------------------------+   messages -------------+
                (loop until 'exit' / Ctrl-C / Ctrl-D)

Run it::

    python 02_continuous_qa/agent.py

Type ``exit`` or ``quit`` (or press Ctrl-C / Ctrl-D) to stop.
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import (  # noqa: E402
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
)

from shared.llm import get_llm  # noqa: E402


SYSTEM_PROMPT = (
    "You are a helpful conversational assistant. Keep answers clear and concise. "
    "Use the conversation history to stay consistent."
)

# Words that end the chat. Kept in one place so it's easy to extend.
EXIT_WORDS = {"exit", "quit", "q"}


def main() -> None:
    """Run an interactive chat loop until the user exits."""
    llm = get_llm()

    # ``history`` is the full transcript. We seed it with the system prompt and
    # then append every user question and model answer. Sending the whole list
    # on each turn is what gives the model "memory" of the conversation.
    history: list[BaseMessage] = [SystemMessage(content=SYSTEM_PROMPT)]

    print("Chat agent ready. Type 'exit' to quit.\n")

    while True:
        # ``input`` can raise EOFError (Ctrl-D) or KeyboardInterrupt (Ctrl-C).
        # We catch both so the program exits cleanly instead of dumping a
        # traceback on the learner.
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        # Ignore blank lines instead of sending an empty request to the API.
        if not user_input:
            continue

        if user_input.lower() in EXIT_WORDS:
            print("Goodbye!")
            break

        # Record the user's turn, then ask the model using the full history.
        history.append(HumanMessage(content=user_input))

        try:
            response = llm.invoke(history)
        except Exception as exc:  # noqa: BLE001 - show a friendly error, keep going
            print(f"[error] {exc}\n")
            # Drop the failed question so a retry starts clean.
            history.pop()
            continue

        answer = response.content
        print(f"Bot: {answer}\n")

        # Record the model's turn so it is remembered next time around.
        history.append(AIMessage(content=answer))


if __name__ == "__main__":
    main()
