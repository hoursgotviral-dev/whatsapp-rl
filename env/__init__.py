"""
env/__init__.py - BULLETPROOF VERSION
"""
from __future__ import annotations

# CORRECT imports for YOUR files
from .environment import WhatsAppEnv, TaskConfig, TASK_CONFIGS
from .simulator.user_simulator import default_simulator

def make_env(task_id: str = "medium", config: TaskConfig | None = None) -> WhatsAppEnv:
    _TASK_MAP = {"task1": "easy", "task2": "medium", "task3": "hard"}
    resolved_id = _TASK_MAP.get(task_id, task_id)
    return WhatsAppEnv(
        task_id=resolved_id, 
        config=config, 
        simulator=default_simulator
    )

__all__ = ["make_env", "WhatsAppEnv", "TaskConfig", "TASK_CONFIGS"]