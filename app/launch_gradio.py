"""Launch wrapper for Gradio on hosted environments.

Keeps gradio_demo.py untouched while handling the
"localhost is not accessible" runtime check seen on some proxies.
"""

from __future__ import annotations

import os
import runpy

import gradio as gr


_original_launch = gr.Blocks.launch


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


gr.Blocks.launch = _patched_launch

runpy.run_path(
    os.path.join(os.path.dirname(__file__), "gradio_demo.py"),
    run_name="__main__",
)

