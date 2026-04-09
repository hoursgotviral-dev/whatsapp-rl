"""Launch wrapper for Gradio on hosted environments.

Keeps gradio_demo.py untouched while handling:
1) localhost accessibility checks behind proxy setups
2) history format compatibility (dict messages -> tuple pairs)
"""

from __future__ import annotations

import os
import runpy
from typing import Any, List

import gradio as gr
import gradio.networking as gr_networking

try:
    from gradio.components.chatbot import Chatbot as ChatbotComponent
except Exception:  # pragma: no cover
    ChatbotComponent = None


_original_launch = gr.Blocks.launch
_original_url_ok = gr_networking.url_ok
_original_postprocess = (
    ChatbotComponent.postprocess if ChatbotComponent is not None else None
)


def _is_hosted_space() -> bool:
    return bool(os.getenv("SPACE_ID") or os.getenv("HF_SPACE_ID") or os.getenv("SPACE_HOST"))


def _patched_url_ok(url: str) -> bool:
    # In hosted proxy setups, localhost probes can fail even when app is healthy.
    if _is_hosted_space():
        return True
    return _original_url_ok(url)


def _append_pair(pairs: List[List[Any]], pending_user: Any, assistant: Any) -> Any:
    if pending_user is None:
        pairs.append([None, assistant])
        return None
    pairs.append([pending_user, assistant])
    return None


def _normalize_history_to_tuples(history: List[Any]) -> List[List[Any]]:
    """
    Convert mixed chatbot history shapes to Gradio tuple format:
    [[user_msg, assistant_msg], ...]
    """
    pairs: List[List[Any]] = []
    pending_user: Any = None

    for item in history:
        role = None
        content = None

        if isinstance(item, dict):
            role = item.get("role")
            content = item.get("content")
        elif hasattr(item, "role") and hasattr(item, "content"):
            role = getattr(item, "role")
            content = getattr(item, "content")
        elif isinstance(item, (list, tuple)):
            if len(item) == 2:
                if pending_user is not None:
                    pairs.append([pending_user, None])
                    pending_user = None
                pairs.append([item[0], item[1]])
                continue
            if len(item) == 1:
                role = "assistant"
                content = item[0]
        else:
            role = "assistant"
            content = item

        if role == "user":
            if pending_user is not None:
                pairs.append([pending_user, None])
            pending_user = content
        else:
            pending_user = _append_pair(pairs, pending_user, content)

    if pending_user is not None:
        pairs.append([pending_user, None])

    return pairs


def _patched_postprocess(self, value, *args, **kwargs):
    # Keep original UI mode; adapt payload only when needed for tuple format.
    try:
        if isinstance(value, list) and value and all(isinstance(v, dict) for v in value):
            value = _normalize_history_to_tuples(value)
        return _original_postprocess(self, value, *args, **kwargs)
    except Exception as exc:
        if "tuples format" in str(exc).lower() and isinstance(value, list):
            normalized = _normalize_history_to_tuples(value)
            return _original_postprocess(self, normalized, *args, **kwargs)
        raise


def _patched_launch(self, *args, **kwargs):
    try:
        return _original_launch(self, *args, **kwargs)
    except ValueError as exc:
        msg = str(exc).lower()
        if "localhost" in msg and "accessible" in msg:
            retry_kwargs = dict(kwargs)
            retry_kwargs["share"] = True
            print("[launch_gradio] Retrying with share=True after localhost probe failure.", flush=True)
            return _original_launch(self, *args, **retry_kwargs)
        raise


gr_networking.url_ok = _patched_url_ok
if ChatbotComponent is not None and _original_postprocess is not None:
    ChatbotComponent.postprocess = _patched_postprocess
gr.Blocks.launch = _patched_launch

runpy.run_path(
    os.path.join(os.path.dirname(__file__), "gradio_demo.py"),
    run_name="__main__",
)
