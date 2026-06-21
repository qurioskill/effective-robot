"""A single factory for building the chat model.

Every example calls :func:`get_llm` instead of constructing ``ChatOpenAI``
directly. That keeps model settings consistent and gives us one obvious place
to tweak temperature, timeouts, retries, etc.
"""

from __future__ import annotations

from langchain_openai import ChatOpenAI


from shared.config import DEFAULT_MODEL, require_api_key


def get_llm(temperature: float = 0.0, model: str | None = "gpt-5.4-mini") -> ChatOpenAI:
    """Build a configured ``ChatOpenAI`` instance.

    Args:
        temperature: Sampling temperature. ``0.0`` gives the most deterministic
            answers, which is usually what you want for Q/A and tool-calling
            demos. Raise it for more creative output.
        model: Optional model name override. Defaults to
            :data:`shared.config.DEFAULT_MODEL`.

    Returns:
        A ready-to-use ``ChatOpenAI`` chat model.
    """
    # Fail fast with a friendly message if the key is missing.
    require_api_key()

    return ChatOpenAI(
        model=model or DEFAULT_MODEL,
        temperature=temperature,
        # Retry transient network/rate-limit errors a couple of times. This is
        # the kind of small touch that makes example code "production ready"
        # without adding complexity.
        max_retries=2,
        timeout=30,
    )
