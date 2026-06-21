# 02 — Continuous Q/A (chat loop with memory)

Example 01, wrapped in a loop. Now you can have a real conversation, and the
agent remembers what was said earlier in the session.

## What this teaches
- A simple **REPL** (read-eval-print loop) around `llm.invoke`.
- **Conversation history**: appending `HumanMessage` / `AIMessage` to a list and
  sending the whole list each turn is what gives the model memory.
- **Graceful exit & error handling**: `exit`/`quit`, Ctrl-C, Ctrl-D, blank
  input, and API errors are all handled without crashing.

## Run it
```bash
python 02_continuous_qa/agent.py
```
Type `exit` (or Ctrl-C / Ctrl-D) to quit.

## What's next
So far the agent can only *talk*. Example **03** gives it a **tool** so it can
*act* — fetching live weather data from the Open-Meteo API.
