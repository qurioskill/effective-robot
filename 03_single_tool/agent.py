"""Example 3 — An agent that can call a tool (live weather via Open-Meteo).

This is the leap from "chatbot" to "agent": the model can now decide to call a
*tool* (a Python function) to fetch information it doesn't have, then use the
result to answer.

We implement the tool-calling loop **by hand** here so you can see exactly what
happens. (Example 04 uses LangChain's prebuilt executor to do this for you.)

Flow
----

    user question
         |
         v
    +-----------+   "I should call get_weather('Paris')"
    |  ChatLLM  | -------------------------------------+
    +-----------+                                      |
         ^                                             v
         |                                     +----------------+
         |   tool result (weather string)      |  get_weather   |  --> Open-Meteo
         +-------------------------------------|     tool       |
                                               +----------------+
         |
         v
    +-----------+
    |  ChatLLM  |  --> final natural-language answer
    +-----------+

Run it::

    python 03_single_tool/agent.py
    python 03_single_tool/agent.py "What's the weather in Tokyo right now?"
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.messages import (  # noqa: E402
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from shared.llm import get_llm  # noqa: E402

# Import the tool from this folder. The sys.path insert above makes the project
# root importable, but the tool lives next to us, so a direct import works.
from tools import get_weather  # noqa: E402


SYSTEM_PROMPT = (
    "You are a helpful weather assistant. "
    "Use the get_weather tool whenever the user asks about weather. "
    "Then answer the user in a friendly, natural sentence."
)

# Tools the model is allowed to call, indexed by name for easy lookup when a
# tool call comes back.
TOOLS = {"get_weather": get_weather}


def run(question: str) -> str:
    """Answer a question, calling the weather tool if the model decides to.

    Args:
        question: The user's question.

    Returns:
        The model's final natural-language answer.
    """
    # ``bind_tools`` tells the model which functions it may call. The model
    # never runs the code itself — it only *requests* a call, which we execute.
    llm = get_llm().bind_tools(list(TOOLS.values()))

    messages: list[BaseMessage] = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=question),
    ]

    # The tool-calling loop. Most weather questions resolve in one tool call,
    # but looping makes the agent robust if the model wants several. We cap the
    # iterations so a misbehaving model can never loop forever.
    for _ in range(5):
        ai_message = llm.invoke(messages)
        messages.append(ai_message)

        # No tool calls means the model produced a final answer.
        if not ai_message.tool_calls:
            return ai_message.content

        # Otherwise, run each requested tool and feed the results back.
        for call in ai_message.tool_calls:
            tool_fn = TOOLS[call["name"]]
            # Show the tool call so the audience can see the agent decide to
            # *act* (call a function) rather than just answer from memory.
            print(f"  -> Tool call: {call['name']}({call['args']})")
            result = tool_fn.invoke(call["args"])
            print(f"  <- Tool result: {result}")
            # A ToolMessage carries the result back to the model, linked to the
            # specific call via tool_call_id.
            messages.append(
                ToolMessage(content=str(result), tool_call_id=call["id"])
            )

    return "Sorry, I couldn't complete that request."


def main() -> None:
    """Entry point: continuously prompt for a city and show the weather.

    Like Example 02, this runs in a loop so the class can try several cities in
    one session. Each round is staged so the audience can see every phase:
      1. the user types a city,
      2. the agent prints the tool call it decides to make (see ``run``),
      3. the agent prints the final weather answer.

    Type 'exit' or 'quit' (or press Ctrl-C / Ctrl-D) to stop.
    """
    print("Weather agent ready. Enter a city, or 'exit' to quit.\n")

    while True:
        # Ask the user which city they want. Handle Ctrl-C / Ctrl-D so the demo
        # never dumps a traceback.
        try:
            city = input("Enter a city: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        # Ignore blank input instead of calling the API with nothing.
        if not city:
            continue

        if city.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        # Turn the city into a natural-language question for the model. The
        # model will then decide to call get_weather(city) — that tool call is
        # printed inside ``run`` so the audience can watch it happen.
        question = f"What's the weather in {city} right now?"

        print("\nThinking...")
        answer = run(question)
        print(f"\nWeather: {answer}\n")


if __name__ == "__main__":
    main()
