"""Launch wrapper for Gradio on hosted environments.

Keeps gradio_demo.py untouched while handling the
"localhost is not accessible" runtime check seen on some proxies.
"""

from __future__ import annotations
import os
import runpy

import gradio as gr

try:
    from gradio.components.chatbot import Chatbot as ChatbotComponent
except Exception:  # pragma: no cover - best-effort compatibility
    ChatbotComponent = None


_original_launch = gr.Blocks.launch
_original_chatbot_init = gr.Chatbot.__init__
_original_component_chatbot_init = (
    ChatbotComponent.__init__ if ChatbotComponent is not None else None
)


def _patched_chatbot_init(self, *args, **kwargs):
    # Gradio 5 can default Chatbot to tuple format.
    # Our app returns OpenAI-style message dicts, so force messages mode.
    if "type" not in kwargs:
        kwargs = dict(kwargs)
        kwargs["type"] = "messages"
    try:
        return _original_chatbot_init(self, *args, **kwargs)
    except TypeError:
        # Safety fallback for unexpected constructor signatures.
        kwargs = dict(kwargs)
        kwargs.pop("type", None)
        return _original_chatbot_init(self, *args, **kwargs)


def _patched_component_chatbot_init(self, *args, **kwargs):
    if _original_component_chatbot_init is None:
        raise RuntimeError("Chatbot component init patch unavailable.")

    if "type" not in kwargs:
        kwargs = dict(kwargs)
        kwargs["type"] = "messages"
    try:
        return _original_component_chatbot_init(self, *args, **kwargs)
    except TypeError:
        kwargs = dict(kwargs)
        kwargs.pop("type", None)
        return _original_component_chatbot_init(self, *args, **kwargs)


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


gr.Chatbot.__init__ = _patched_chatbot_init
if ChatbotComponent is not None:
    ChatbotComponent.__init__ = _patched_component_chatbot_init
gr.Blocks.launch = _patched_launch

runpy.run_path(
    os.path.join(os.path.dirname(__file__), "gradio_demo.py"),
    run_name="__main__",
)
