"""Launch wrapper for Gradio on hosted environments.

Keeps gradio_demo.py untouched while handling:
1) localhost accessibility checks on hosted proxies
2) history format compatibility (dict messages -> tuple pairs)
"""

from __future__ import annotations

import os
import runpy
from typing import Any, List

import gradio as gr

try:
    from gradio.components.chatbot import Chatbot as ChatbotComponent
except Exception:  # pragma: no cover
    ChatbotComponent = None


_original_launch = gr.Blocks.launch
_original_postprocess = (
    ChatbotComponent.postprocess if ChatbotComponent is not None else None
)


def _messages_to_tuples(history: List[Any]) -> List[List[Any]]:
    """Convert OpenAI-style message dict history to tuple chatbot history."""
    tuples: List[List[Any]] = []
    pending_user: Any = None

    for item in history:
        if not isinstance(item, dict):
            continue
        role = item.get("role")
        content = item.get("content")

        if role == "user":
            if pending_user is not None:
                tuples.append([pending_user, None])
            pending_user = content
        else:
            if pending_user is None:
                tuples.append([None, content])
            else:
                tuples.append([pending_user, content])
                pending_user = None

    if pending_user is not None:
        tuples.append([pending_user, None])

    return tuples


def _patched_postprocess(self, value, *args, **kwargs):
    # Keep original UI mode; only adapt data format when callbacks emit dict messages.
    if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
        value = _messages_to_tuples(value)
    return _original_postprocess(self, value, *args, **kwargs)


def _patched_launch(self, *args, **kwargs):
    try:
        return _original_launch(self, *args, **kwargs)
    except ValueError as exc:
        message = str(exc)
        if "localhost is not accessible" not in message:
            raise

        retry_kwargs = dict(kwargs)
        retry_kwargs["share"] = True
        print(
            "[launch_gradio] Retrying launch with share=True due localhost check.",
            flush=True,
        )
        return _original_launch(self, *args, **retry_kwargs)


if ChatbotComponent is not None and _original_postprocess is not None:
    ChatbotComponent.postprocess = _patched_postprocess

gr.Blocks.launch = _patched_launch

runpy.run_path(
    os.path.join(os.path.dirname(__file__), "gradio_demo.py"),
    run_name="__main__",
)