# 04 — Agent with multiple tools (automatic chaining)

Three tools, and the agent decides which to use and in what order. Instead of
the hand-written loop from Example 03, we use LangChain's prebuilt
`create_agent` (LangChain 1.x), which runs the tool loop for us.

## Tools
- `geocode_city(city)` — city name → latitude/longitude
- `get_weather(latitude, longitude)` — coordinates → current weather
- `get_current_time(timezone)` — IANA timezone → local time

Splitting geocoding from weather forces the agent to **chain** calls:
`geocode_city → get_weather`.

## What this teaches
- `create_agent` (the tool-calling loop, handled for you).
- Driving it with a `messages` list and reading the full transcript back.
- Multi-step tool **chaining** — the model orders the calls itself.
- Walking the returned messages to print each tool call live.

## Run it
```bash
python 04_multi_tool/agent.py
```
It runs in a loop (like Examples 02 and 03), prompting for a question each
round. Each tool call and result is printed so you can watch the agent chain
tools. Type `exit` (or Ctrl-C / Ctrl-D) to quit.
```
You: What's the weather in Berlin and what time is it there?
You: Compare the weather in Tokyo and Sydney
You: exit
```

## What's next
One agent with many tools has limits. Example **05** splits the work across
*multiple specialist agents* coordinated by a supervisor (LangGraph).
