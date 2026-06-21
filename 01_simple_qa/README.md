# 01 — Simple Q/A (runs once)

The smallest useful LLM program: one question in, one answer out, then exit.

## What this teaches
- How to construct a chat model (`ChatOpenAI`) via the shared factory.
- The **system message + human message** pattern.
- `llm.invoke(messages)` for a single request/response round-trip.

## Run it
```bash
python 01_simple_qa/agent.py
python 01_simple_qa/agent.py "What is retrieval-augmented generation?"
```

## What's next
Example **02** wraps this exact idea in a loop so you can have a back-and-forth
conversation instead of asking a single question.
