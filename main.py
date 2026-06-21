"""Entry hint for the teaching repo.

There is no single "main" program — each of the 5 examples is run directly from
its own folder. This file just points you to them.

    python 01_simple_qa/agent.py
    python 02_continuous_qa/agent.py
    python 03_single_tool/agent.py
    python 04_multi_tool/agent.py
    python 05_multi_agent/agent.py

See README.md for setup and details.
"""

EXAMPLES = [
    ("01_simple_qa", "Ask once, answer once, exit."),
    ("02_continuous_qa", "Chat loop with memory."),
    ("03_single_tool", "Agent with one tool (live weather)."),
    ("04_multi_tool", "Agent that chains multiple tools."),
    ("05_multi_agent", "Supervisor + specialist agents (LangGraph)."),
]


def main() -> None:
    print("AI Agents from Scratch — run an example directly:\n")
    for folder, desc in EXAMPLES:
        print(f"  python {folder}/agent.py   # {desc}")
    print("\nSee README.md for setup instructions.")


if __name__ == "__main__":
    main()


import requests

url = "http://localhost:11434/api/chat"

payload = {
    "model": "llama3.2",
    "messages": [
        {
            "role": "user",
            "content": "Explain evaporation in simple terms."
        }
    ],
    "stream": False
}

response = requests.post(url, json=payload)
response.raise_for_status()

data = response.json()
print(data["message"]["content"])