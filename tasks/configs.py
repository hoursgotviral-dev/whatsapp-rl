"""
Task configurations for the WhatsApp Business RL environment.

Each TaskConfig defines:
- Episode length and user type distributions.
- Initial state variable ranges.
- Reward weight overrides.
"""

from dataclasses import dataclass, field
from typing import Dict, Tuple


@dataclass(frozen=True, kw_only=True)
class TaskConfig:
    """Configuration for a single task."""
    
    task_id: str
    """Unique identifier for the task."""
    
    max_steps: int
    """Maximum episode length."""
    
    user_type_probs: Dict[str, float]
    """Probability distribution over user types."""
    
    initial_state_ranges: Dict[str, Tuple[float, float]]
    """Initial ranges for latent state variables."""
    
    reward_weights: Dict[str, float] = field(default_factory=dict)
    """Optional overrides for reward component weights."""
    
    description: str = ""
    """Human-readable description of the task."""


TASK1_CONFIG = TaskConfig(
    task_id="task1",
    max_steps=8,
    user_type_probs={
        "price_sensitive": 0.30,
        "busy": 0.30,
        "indecisive": 0.20,
        "impulsive": 0.20,
    },
    initial_state_ranges={
        "trust": (0.4, 0.7),
        "patience": (0.5, 0.8),
        "annoyance": (0.0, 0.2),
        "satisfaction": (0.4, 0.7),
        "conversion_prob": (0.2, 0.5),
        "cost_to_business": (0.0, 0.0),
    },
    reward_weights={
        "final_conversion": 1.0,
        "churn_penalty": -1.0,
        "step_engagement": 0.05,
    },
    description="Short horizon, simple inquiry handling, single product focus.",
)


TASK2_CONFIG = TaskConfig(
    task_id="task2",
    max_steps=12,
    user_type_probs={
        "price_sensitive": 0.35,
        "busy": 0.20,
        "indecisive": 0.30,
        "impulsive": 0.15,
    },
    initial_state_ranges={
        "trust": (0.3, 0.6),
        "patience": (0.4, 0.7),
        "annoyance": (0.0, 0.2),
        "satisfaction": (0.3, 0.6),
        "conversion_prob": (0.15, 0.45),
        "cost_to_business": (0.0, 0.0),
    },
    reward_weights={
        "final_conversion": 1.5,
        "churn_penalty": -1.2,
        "step_engagement": 0.04,
        "obligation_complete": 0.1,
    },
    description="Multi-stage sales with negotiation and key info collection.",
)


TASK3_CONFIG = TaskConfig(
    task_id="task3",
    max_steps=16,
    user_type_probs={
        "price_sensitive": 0.30,
        "busy": 0.25,
        "indecisive": 0.25,
        "impulsive": 0.20,
    },
    initial_state_ranges={
        "trust": (0.2, 0.6),
        "patience": (0.3, 0.7),
        "annoyance": (0.0, 0.3),
        "satisfaction": (0.2, 0.6),
        "conversion_prob": (0.1, 0.4),
        "cost_to_business": (0.0, 0.0),
    },
    reward_weights={
        "final_conversion": 1.5,
        "churn_penalty": -1.5,
        "step_engagement": 0.03,
        "obligation_complete": 0.15,
        "obligation_expire": -0.3,
    },
    description="Longer episodes, mixed user types, more obligations and follow-ups.",
)


TASK_CONFIGS = {
    "task1": TASK1_CONFIG,
    "task2": TASK2_CONFIG,
    "task3": TASK3_CONFIG,
}


def get_task_config(task_id: str) -> TaskConfig:
    """
    Get TaskConfig by ID.
    
    Raises KeyError if task_id not found.
    """
    if task_id not in TASK_CONFIGS:
        available = list(TASK_CONFIGS.keys())
        raise KeyError(
            f"Unknown task_id '{task_id}'. "
            f"Available tasks: {available}"
        )
    return TASK_CONFIGS[task_id]


    # OpenEnv compatibility export
def get_openenv_config(task_id: str) -> dict:
    """
    Returns plain dict for OpenEnv server (not TaskConfig object).
    Compatible with make_env(task_id, config_dict).
    """
    config = get_task_config(task_id)
    return {
        "task_id": config.task_id,
        "max_steps": config.max_steps,
        "user_type_probs": config.user_type_probs,
        "initial_state_ranges": config.initial_state_ranges,
        "reward_weights": config.reward_weights,
        "description": config.description,
    }