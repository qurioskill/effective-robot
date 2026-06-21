# 03 — Agent with a single tool (live weather)

The jump from chatbot to **agent**: the model can now call a Python function
(a "tool") to fetch live data and use the result in its answer.

We use the free, no-key [Open-Meteo](https://open-meteo.com/) API for weather.

## What this teaches
- Defining a tool with the `@tool` decorator (the docstring + type hints become
  the schema the model reads).
- `llm.bind_tools(...)` to expose tools to the model.
- The **tool-calling loop**, written out by hand:
  1. model requests a tool call,
  2. we execute the function,
  3. we send the result back as a `ToolMessage`,
  4. the model writes the final answer.

## Run it
```bash
python 03_single_tool/agent.py
```
It runs in a loop (like Example 02), prompting for a city each round and showing
three staged steps. Type `exit` (or Ctrl-C / Ctrl-D) to quit.
```
Enter a city: Tokyo

Thinking...
  -> Tool call: get_weather({'city': 'Tokyo'})
  <- Tool result: Current weather in Tokyo, Japan: partly cloudy, 21°C, wind 9 km/h.

Weather: It's currently partly cloudy in Tokyo, around 21°C with a light breeze.

Enter a city: exit
Goodbye!
```

## What's next
Example **04** adds *more* tools and lets LangChain's prebuilt agent executor
run the loop automatically, so the model can chain several tools together.
