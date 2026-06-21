"""Example 4 — An agent with multiple tools, run by LangChain's create_agent.

In Example 03 we wrote the tool-calling loop by hand. Here we hand that job to
LangChain's prebuilt **create_agent** (LangChain 1.x), which automatically:

    * sends the question to the model,
    * runs any tools the model requests,
    * feeds results back,
    * repeats until the model gives a final answer.

Because there are now three tools, the model often has to *chain* them, e.g.:

    "What's the weather in Berlin and what time is it there?"
        -> geocode_city("Berlin")        (get coordinates)
        -> get_weather(lat, lon)         (get conditions)
        -> get_current_time("Europe/...")(get local time)
        -> final answer combining all three

Flow
----

    question --> +----------------+   picks & runs tools in a loop
                 |  create_agent  | <----> [ geocode_city ]
                 |  (LangChain)   | <----> [ get_weather  ]
                 |                | <----> [ get_current_time ]
                 +----------------+
                        |
                        v
                  final answer

Run it::

    python 04_multi_tool/agent.py
    python 04_multi_tool/agent.py "Weather and local time in Berlin?"
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain.agents import create_agent  # noqa: E402
from langchain_core.messages import AIMessage, ToolMessage  # noqa: E402

from shared.llm import get_llm  # noqa: E402

from tools import geocode_city, get_current_time, get_weather  # noqa: E402


TOOLS = [geocode_city, get_weather, get_current_time]

# The system prompt steers the agent's tool use. We tell it explicitly to
# geocode before fetching weather, which is what produces the multi-step chain.
SYSTEM_PROMPT = (
    "You are a helpful assistant with weather and time tools. "
    "To get weather for a city you must first geocode it to get coordinates, "
    "then call get_weather with those coordinates. "
    "Answer the user in a friendly, natural sentence."
)


def build_agent():
    """Construct the agent with our tools.

    ``create_agent`` (LangChain 1.x) returns a ready-to-run agent that handles
    the tool-calling loop for us — it sends the question to the model, runs any
    tools the model requests, feeds results back, and repeats until done.

    Returns:
        A runnable agent (a compiled LangGraph graph under the hood).
    """
    return create_agent(get_llm(), TOOLS, system_prompt=SYSTEM_PROMPT)


def print_tool_steps(messages: list) -> None:
    """Print each tool call and its result so the class can watch the chain.

    ``create_agent`` doesn't have the old ``verbose=True`` flag, so we walk the
    returned message list ourselves and surface the tool activity.
    """
    for msg in messages:
        # An AIMessage may carry one or more tool calls the model decided to make.
        if isinstance(msg, AIMessage) and msg.tool_calls:
            for call in msg.tool_calls:
                print(f"  -> Tool call: {call['name']}({call['args']})")
        # A ToolMessage carries the result of one of those calls.
        elif isinstance(msg, ToolMessage):
            print(f"  <- Tool result: {msg.content}")


def main() -> None:
    """Entry point: continuously prompt for a question and answer it.

    Like Examples 02 and 03, this runs in a loop so the class can try several
    questions in one session. We print each tool call the agent makes, so the
    audience can watch it chain tools together.

    Type 'exit' or 'quit' (or press Ctrl-C / Ctrl-D) to stop.
    """
    # Build the agent once and reuse it for every question.
    agent = build_agent()

    print("Multi-tool agent ready. Ask a question, or 'exit' to quit.\n")

    while True:
        # Handle Ctrl-C / Ctrl-D so the demo never dumps a traceback.
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        # Ignore blank input instead of invoking the agent with nothing.
        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        # ``create_agent`` takes a messages list and returns the full transcript
        # (the user turn, the model's tool calls, the tool results, and the
        # final answer) under the "messages" key.
        result = agent.invoke({"messages": [("user", question)]})
        print_tool_steps(result["messages"])

        final_answer = result["messages"][-1].content
        print("\n" + "=" * 60)
        print(f"A: {final_answer}\n")


if __name__ == "__main__":
    main()
