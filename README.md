# AI Agents from Scratch (Python)

A hands-on teaching repo that builds **5 kinds of AI agents** in Python, from a
one-shot Q/A bot up to a multi-agent team. Each example lives in its own folder
and builds on the previous one.

Built with **Python + OpenAI API + LangChain** (and **LangGraph** for the
multi-agent example). Weather demos use the free, no-key
[Open-Meteo](https://open-meteo.com/) API.

## The 5 examples

| # | Folder | Concept |
|---|--------|---------|
| 1 | [`01_simple_qa`](01_simple_qa/) | Ask once, answer once, exit. |
| 2 | [`02_continuous_qa`](02_continuous_qa/) | The same bot in a chat loop with memory. |
| 3 | [`03_single_tool`](03_single_tool/) | An agent that calls **one tool** (live weather). |
| 4 | [`04_multi_tool`](04_multi_tool/) | An agent that **chains multiple tools**. |
| 5 | [`05_multi_agent`](05_multi_agent/) | A **supervisor + specialists** team (LangGraph). |

Each folder has its own `README.md` explaining what's new versus the previous
step.

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Add your OpenAI API key
cp .env.example .env
# then edit .env and paste your key from https://platform.openai.com/api-keys
```

## Running an example

Run any example from the **project root**:

```bash
python 01_simple_qa/agent.py
python 02_continuous_qa/agent.py
python 03_single_tool/agent.py "What's the weather in Tokyo?"
python 04_multi_tool/agent.py  "Weather and local time in Berlin?"
python 05_multi_agent/agent.py "Weather in Rome, and who built the Colosseum?"
```

Most examples accept an optional question as a command-line argument and fall
back to a sensible default if you don't provide one.

## Suggested teaching order

Walk through the folders 1 → 5. Each step introduces exactly one new idea:

1. **Calling an LLM** (messages in, answer out)
2. **Conversation loops + memory**
3. **Tools** — letting the model *act*, not just talk
4. **Multiple tools + automatic chaining**
5. **Multiple agents** coordinated by a supervisor

## Project layout

```
.
├── README.md              <- you are here
├── requirements.txt
├── .env.example
├── shared/                <- config + LLM factory used by every example
│   ├── config.py
│   └── llm.py
├── 01_simple_qa/
├── 02_continuous_qa/
├── 03_single_tool/
├── 04_multi_tool/
└── 05_multi_agent/
```
