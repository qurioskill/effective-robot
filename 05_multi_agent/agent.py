"""Example 5 — A multi-agent system with a supervisor (LangGraph).

Instead of one agent juggling everything, we use a small *team*:

    * supervisor    - reads the conversation and decides who works next.
    * weather_agent - a specialist that can call the weather tool.
    * general_agent - a specialist that answers general-knowledge questions.

The supervisor routes each turn to one specialist (or decides the work is done
and finishes). This "supervisor + workers" pattern scales far better than
cramming every tool and instruction into a single prompt.

Graph
-----

                +--------------+
        +-----> |  supervisor  | ----> FINISH ----> done
        |       +--------------+
        |          |        |
        |  weather |        | general
        |          v        v
        |  +--------------+  +--------------+
        +--| weather_agent|  | general_agent|--+
           +--------------+  +--------------+  |
                  ^                            |
                  +----------------------------+
            (each worker reports back to the supervisor)

We build this with LangGraph, whose ``StateGraph`` lets us define nodes (the
agents) and edges (who runs after whom) explicitly.

Run it (loops until you type 'exit')::

    python 05_multi_agent/agent.py
"""

from __future__ import annotations

import os
import sys
from typing import Annotated, Literal, TypedDict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage  # noqa: E402
from langchain_core.tools import tool  # noqa: E402
from langgraph.graph import END, START, StateGraph  # noqa: E402
from langgraph.graph.message import add_messages  # noqa: E402
from langgraph.prebuilt import create_react_agent  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from shared.llm import get_llm  # noqa: E402


# --------------------------------------------------------------------------- #
# Tool: a self-contained weather lookup (city name -> current weather).
# --------------------------------------------------------------------------- #
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city by name (via Open-Meteo)."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1},
            timeout=10,
        )
        geo.raise_for_status()
        results = geo.json().get("results")
        if not results:
            return f"No city found named '{city}'."
        place = results[0]
        forecast = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current_weather": True,
            },
            timeout=10,
        )
        forecast.raise_for_status()
        current = forecast.json().get("current_weather", {})
    except requests.RequestException as exc:
        return f"Weather service error: {exc}"

    return (
        f"Weather in {place['name']}: {current.get('temperature')}°C, "
        f"wind {current.get('windspeed')} km/h."
    )


# --------------------------------------------------------------------------- #
# Shared graph state. Every node receives and returns this structure.
# --------------------------------------------------------------------------- #
class TeamState(TypedDict):
    """State passed between nodes.

    ``messages`` is the running conversation. The ``add_messages`` reducer means
    each node *appends* to the list instead of overwriting it.
    ``next`` is set by the supervisor to name the node that should run next.
    """

    messages: Annotated[list[BaseMessage], add_messages]
    next: str


# The specialists the supervisor can choose from, plus FINISH to stop.
MEMBERS = ["weather_agent", "general_agent"]
ROUTING_OPTIONS = MEMBERS + ["FINISH"]


class Route(BaseModel):
    """Structured decision returned by the supervisor."""

    next: Literal["weather_agent", "general_agent", "FINISH"] = Field(
        description="Which specialist should act next, or FINISH if the user's "
        "question has been fully answered."
    )


# Build the specialist agents once at import time. ``create_react_agent`` returns
# a runnable graph that handles its own internal tool-calling loop.
_weather_agent = create_react_agent(
    get_llm(),
    tools=[get_weather],
    prompt="You are a weather specialist. Use the get_weather tool to answer "
    "weather questions, then reply with a short, friendly sentence.",
)
_general_agent = create_react_agent(
    get_llm(),
    tools=[],  # no tools — answers from its own knowledge
    prompt="You are a general-knowledge specialist. Answer the user's "
    "non-weather question concisely.",
)


# --------------------------------------------------------------------------- #
# Nodes.
# --------------------------------------------------------------------------- #
def supervisor_node(state: TeamState) -> dict:
    """Decide which specialist should act next (or finish).

    The supervisor looks at the whole conversation and uses structured output to
    return exactly one of the routing options.
    """
    system = (
        "You are a supervisor routing work between these specialists: "
        f"{MEMBERS}. Given the conversation, decide who should act next. "
        "Route weather questions to weather_agent and everything else to "
        "general_agent. When every part of the user's request has an answer, "
        "respond with FINISH."
    )
    llm = get_llm().with_structured_output(Route)
    decision: Route = llm.invoke(
        [HumanMessage(content=system)] + state["messages"]
    )
    print(f"  [supervisor] -> {decision.next}")
    return {"next": decision.next}


def weather_node(state: TeamState) -> dict:
    """Run the weather specialist and append its answer to the conversation."""
    result = _weather_agent.invoke({"messages": state["messages"]})
    # The specialist's last message is its answer. We re-label it so the
    # supervisor can see which agent produced it.
    answer = result["messages"][-1].content
    return {"messages": [AIMessage(content=answer, name="weather_agent")]}


def general_node(state: TeamState) -> dict:
    """Run the general-knowledge specialist and append its answer."""
    result = _general_agent.invoke({"messages": state["messages"]})
    answer = result["messages"][-1].content
    return {"messages": [AIMessage(content=answer, name="general_agent")]}


def route_from_supervisor(state: TeamState) -> Literal["weather_agent", "general_agent", "__end__"]:
    """Translate the supervisor's decision into the next graph edge."""
    if state["next"] == "FINISH":
        return END
    return state["next"]


def build_graph():
    """Wire the nodes and edges into a runnable LangGraph application."""
    graph = StateGraph(TeamState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("weather_agent", weather_node)
    graph.add_node("general_agent", general_node)

    # Always start at the supervisor.
    graph.add_edge(START, "supervisor")

    # After the supervisor, branch to the chosen specialist or end.
    graph.add_conditional_edges(
        "supervisor",
        route_from_supervisor,
        {"weather_agent": "weather_agent", "general_agent": "general_agent", END: END},
    )

    # After a specialist runs, return control to the supervisor so it can
    # decide whether more work is needed.
    graph.add_edge("weather_agent", "supervisor")
    graph.add_edge("general_agent", "supervisor")

    return graph.compile()


def main() -> None:
    """Entry point: continuously prompt for a question and run the team.

    Like Examples 02-04, this runs in a loop so the class can try several
    questions in one session. Each question starts with fresh state, so the
    supervisor routes it independently. The supervisor prints its routing
    decisions so the audience can watch the team coordinate.

    Type 'exit' or 'quit' (or press Ctrl-C / Ctrl-D) to stop.
    """
    # Build the graph once and reuse it for every question.
    app = build_graph()

    print("Multi-agent team ready. Ask a question, or 'exit' to quit.\n")

    while True:
        # Handle Ctrl-C / Ctrl-D so the demo never dumps a traceback.
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        # Ignore blank input instead of running the team with nothing.
        if not question:
            continue

        if question.lower() in {"exit", "quit", "q"}:
            print("Goodbye!")
            break

        # ``recursion_limit`` caps total node visits so the team can't loop
        # forever. Each question starts from a clean state.
        final_state = app.invoke(
            {"messages": [HumanMessage(content=question)], "next": ""},
            config={"recursion_limit": 15},
        )

        print("\n" + "=" * 60)
        print("Conversation:")
        for msg in final_state["messages"]:
            speaker = getattr(msg, "name", None) or msg.__class__.__name__
            print(f"  [{speaker}] {msg.content}")
        print()


if __name__ == "__main__":
    main()
