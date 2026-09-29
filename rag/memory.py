"""Sliding-window conversation memory."""

from langchain_core.messages import BaseMessage


def window(messages: list[BaseMessage], n: int) -> list[BaseMessage]:
    """Return the last n messages (n <= 0 means no history)."""
    return list(messages[-n:]) if n > 0 else []
