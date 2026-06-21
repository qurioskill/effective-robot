# 05 — Multi-agent system (supervisor + specialists)

Instead of one agent doing everything, a **supervisor** routes each turn to a
specialist:

- `weather_agent` — has the `get_weather` tool.
- `general_agent` — answers general-knowledge questions (no tools).

The supervisor reads the conversation, picks who works next, and finishes when
the whole request is answered. Built with **LangGraph**.

## Graph
```
        +--------------+
 +----> |  supervisor  | --> FINISH --> done
 |      +--------------+
 |        |         |
 | weather|         | general
 |        v         v
 |  weather_agent   general_agent
 |        |         |
 +--------+---------+   (workers report back to the supervisor)
```

## What this teaches
- LangGraph `StateGraph`: nodes (agents) + edges (control flow).
- A **supervisor** that routes work using structured output (`with_structured_output`).
- Prebuilt `create_react_agent` for each specialist.
- Shared state with the `add_messages` reducer, and a `recursion_limit` guard.

## Run it
```bash
python 05_multi_agent/agent.py
```
It runs in a loop (like Examples 02-04), prompting for a question each round.
The supervisor prints its routing decisions so you can watch the team
coordinate. Type `exit` (or Ctrl-C / Ctrl-D) to quit.
```
You: Weather in Rome, and who built the Colosseum?
You: exit
```

## Where to go next
Add more specialists (a math agent, a search agent), give the supervisor memory,
or let specialists call *each other*. The supervisor pattern scales to all of it.
