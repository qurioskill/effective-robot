"""Central configuration for all agent examples.

Responsibilities
----------------
1. Load environment variables from a local ``.env`` file (if present).
2. Validate that the OpenAI API key is available before any agent runs.
3. Expose a single ``DEFAULT_MODEL`` so every example uses the same model.

Why centralize this?
--------------------
In a live teaching session you do not want to hunt through five different files
to change a model name or debug a missing API key. Everything lives here.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

# ``load_dotenv`` reads a ``.env`` file in the project root and copies its
# values into ``os.environ``. It is a no-op if the file does not exist, so this
# is safe to call unconditionally at import time.
load_dotenv()

# The model used by every example. ``gpt-4o-mini`` is cheap, fast, and supports
# tool calling, which makes it a good default for teaching. Override it by
# setting OPENAI_MODEL in your environment or ``.env`` file.
DEFAULT_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def require_api_key() -> str:
    """Return the OpenAI API key, raising a clear error if it is missing.

    Calling this at the top of each example gives the learner an immediate,
    friendly message instead of a confusing failure deep inside the LangChain
    or OpenAI libraries.

    Returns:
        The API key string.

    Raises:
        RuntimeError: If ``OPENAI_API_KEY`` is not set.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY is not set.\n"
            "  1. Copy .env.example to .env\n"
            "  2. Add your key from https://platform.openai.com/api-keys\n"
        )
    return api_key
