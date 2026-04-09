"""Launch wrapper for Gradio on hosted environments.

Keeps gradio_demo.py untouched while handling the
"localhost is not accessible" runtime check seen on some proxies.
"""

from __future__ import annotations

import inspect
import os
import runpy

import gradio as gr


_original_launch = gr.Blocks.launch
_original_chatbot_init = gr.Chatbot.__init__


def _patched_chatbot_init(self, *args, **kwargs):
    # Gradio 5 defaults Chatbot to tuple format in some versions.
    # Our app returns OpenAI-style message dicts, so ensure compatible mode.
    params = inspect.signature(_original_chatbot_init).parameters
    if "type" in params and "type" not in kwargs:
        kwargs["type"] = "messages"
    return _original_chatbot_init(self, *args, **kwargs)


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
gr.Blocks.launch = _patched_launch

runpy.run_path(
    os.path.join(os.path.dirname(__file__), "gradio_demo.py"),
    run_name="__main__",
)
