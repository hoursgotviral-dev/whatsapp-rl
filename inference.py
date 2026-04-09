"""
inference.py - WhatsApp Sales RL Inference Script
=================================================
Runs one episode per task (task1, task2, task3) using an LLM agent.

Required environment variables:
  API_BASE_URL   - OpenAI-compatible endpoint (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     - Model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       - API key / Hugging Face token

STDOUT FORMAT (one [START], N [STEP]s, one [END] per episode):
  [START] task=<task_id> env=whatsapp_sales_rl model=<model_name>
  [STEP]  step=<n> action=<action_type> reward=<0.00> done=<true|false> error=<msg|null>
  [END]   success=<true|false> steps=<n> rewards=<r1,r2,...,rn>
"""

from __future__ import annotations

import json
import os
import sys
import textwrap
from typing import Any, Dict, List, Optional

from openai import OpenAI

# Make project importable from root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import make_env
from models import Action, Observation

# Environment variables
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME = os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct")
HF_TOKEN = os.getenv("HF_TOKEN")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
LLM_API_KEY = HF_TOKEN or OPENAI_API_KEY
LOCAL_IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME")

BENCHMARK = "whatsapp_sales_rl"
TASKS = ["task1", "task2", "task3"]
TASK_SEEDS = {"task1": 42, "task2": 43, "task3": 44}

TEMPERATURE = 0.0
MAX_TOKENS = 256
REQUEST_TIMEOUT_S = 30

# Treat escalation as a fallback, not a successful sales outcome.
SUCCESS_OUTCOMES = {"SALE"}


def log_start(task: str, model: str) -> None:
    print(f"[START] task={task} env={BENCHMARK} model={model}", flush=True)


def log_step(
    step: int,
    action: str,
    reward: float,
    done: bool,
    error: Optional[str],
) -> None:
    error_val = error if error else "null"
    print(
        f"[STEP] step={step} action={action} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_val}",
        flush=True,
    )


def log_end(success: bool, steps: int, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}",
        flush=True,
    )


SYSTEM_PROMPT = textwrap.dedent(
    """
You are a WhatsApp sales agent. Your goal is to raise the customer's conversion
probability to close a SALE by building trust and satisfaction, not by spamming discounts.

CRITICAL RULES:
1. OFFER_DISCOUNT is expensive. Use it at most once per episode, only when sentiment > 0.2
   and the customer has explicitly resisted the price. Keep discount_pct between 5 and 15.
2. Never repeat the same action twice in a row.
3. DELAY_RESPONSE is forbidden. Never use it.
4. Build trust first: ASK_QUESTION -> PROVIDE_INFO -> GIVE_PRICE -> small OFFER_DISCOUNT -> close.
5. If sentiment is already positive (> 0.1), skip discounts and use PROVIDE_INFO to reinforce.
6. If uncertainties include "low_trust", use PROVIDE_INFO or ASK_QUESTION, not discounts.
7. If uncertainties include "low_patience", wrap up quickly using GIVE_PRICE or PROVIDE_INFO.

Ideal action sequence by stage:
  GREETING           -> ASK_QUESTION
  DISCOVERY          -> ASK_QUESTION once, then PROVIDE_INFO
  QUALIFICATION      -> PROVIDE_INFO or GIVE_PRICE
  OBJECTION_HANDLING -> PROVIDE_INFO first; OFFER_DISCOUNT (5-10%) only if still resistant
  NEGOTIATION        -> OFFER_DISCOUNT once (10-15% max), then PROVIDE_INFO
  CLOSING            -> PROVIDE_INFO to reinforce decision
  POST_SALE          -> PROVIDE_INFO

Reply with a valid JSON object and nothing else. No extra text.
  {"action_type": "<ACTION>", "message": "<your message to the customer>"}

For OFFER_DISCOUNT:
  {"action_type": "OFFER_DISCOUNT", "discount_pct": 10, "message": "<msg>"}

Available actions:
  ASK_QUESTION, GIVE_PRICE, OFFER_DISCOUNT, PROVIDE_INFO,
  ESCALATE, DELAY_RESPONSE, END_CONVERSATION
"""
).strip()


def _build_user_prompt(obs: Observation, step: int) -> str:
    history_text = "\n".join(obs.chat_history[-6:]) if obs.chat_history else "None"
    uncertainties = ", ".join(obs.uncertainties) if obs.uncertainties else "none"
    pending = obs.obligations.pending
    obligations_text = (
        "\n".join(f"  - [{o.type}] {o.description}" for o in pending)
        if pending
        else "  none"
    )

    if "low_patience" in obs.uncertainties:
        hint = "WARNING: Customer patience is low. Wrap up fast using GIVE_PRICE or PROVIDE_INFO."
    elif "low_trust" in obs.uncertainties:
        hint = "WARNING: Trust is low. Do not offer discounts. Use PROVIDE_INFO or ASK_QUESTION."
    elif "high_annoyance" in obs.uncertainties:
        hint = "WARNING: Customer is annoyed. Be concise and use PROVIDE_INFO."
    elif step <= 2:
        hint = "Early stage: ask one focused question to understand needs."
    elif step <= 5:
        hint = "Mid stage: provide value and share product details."
    elif obs.sentiment > 0.1:
        hint = "Sentiment is positive. Reinforce with PROVIDE_INFO and move toward closing."
    else:
        hint = "Late stage: offer a small one-time discount (5-10%) to break resistance."

    return textwrap.dedent(
        f"""
        Step {step} of episode:

        Stage:         {obs.stage}
        Intent:        {obs.intent}
        Sentiment:     {obs.sentiment:+.2f}   (-1=very negative, +1=very positive)
        Uncertainties: {uncertainties}
        Step count:    {obs.step_count}

        Coach advice:  {hint}

        Pending obligations (fulfil with PROVIDE_INFO or ASK_QUESTION):
        {obligations_text}

        Last 6 conversation lines:
        {history_text}

        Reply with JSON only. No explanation.
        """
    ).strip()


def _extract_json_object(raw: str) -> Dict[str, Any]:
    """Extract the first JSON object from a model response."""
    candidate = raw.strip()
    if candidate.startswith("```"):
        parts = candidate.split("```")
        if len(parts) >= 2:
            candidate = parts[1].strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()

    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return {}

    snippet = candidate[start : end + 1]
    try:
        parsed = json.loads(snippet)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def _call_llm(client: Optional[OpenAI], obs: Observation, step: int) -> Dict[str, Any]:
    """
    Ask the LLM for an action. Returns a dict with at minimum 'action_type'.
    Falls back to an empty dict (triggers heuristic) on any error.
    """
    if client is None:
        return {}

    user_prompt = _build_user_prompt(obs, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            timeout=REQUEST_TIMEOUT_S,
        )
        raw = (completion.choices[0].message.content or "").strip()
        return _extract_json_object(raw)
    except Exception as exc:
        print(f"LLM call failed at step {step}: {exc}", file=sys.stderr, flush=True)
        return {}


VALID_ACTIONS = {
    "ASK_QUESTION",
    "GIVE_PRICE",
    "OFFER_DISCOUNT",
    "PROVIDE_INFO",
    "ESCALATE",
    "DELAY_RESPONSE",
    "END_CONVERSATION",
}

_STAGE_FALLBACK = {
    "GREETING": ("ASK_QUESTION", "How can I help you today?"),
    "DISCOVERY": ("ASK_QUESTION", "Could you tell me more about what you're looking for?"),
    "QUALIFICATION": ("PROVIDE_INFO", "Here's what makes our product a great fit for you."),
    "OBJECTION_HANDLING": ("PROVIDE_INFO", "Let me address your concerns directly."),
    "NEGOTIATION": ("GIVE_PRICE", "I can share a final quote to help you decide."),
    "CLOSING": ("PROVIDE_INFO", "You've made a great choice!"),
    "POST_SALE": ("PROVIDE_INFO", "Thank you for your purchase!"),
    "ESCALATED": ("ESCALATE", "Let me connect you with someone senior."),
    "ENDED": ("END_CONVERSATION", "Thank you for your time."),
}

_NO_REPEAT = {
    "ASK_QUESTION": "PROVIDE_INFO",
    "GIVE_PRICE": "PROVIDE_INFO",
    "OFFER_DISCOUNT": "PROVIDE_INFO",
    "ESCALATE": "PROVIDE_INFO",
    "END_CONVERSATION": "PROVIDE_INFO",
}

# Module-level flag: only one discount per episode
_discount_used: bool = False


def _fallback_action_by_state(obs: Observation, last_action_type: str = "") -> Action:
    """Deterministic fallback policy tuned for stable conversion."""
    global _discount_used

    if obs.obligations.has_pending:
        action_type = "PROVIDE_INFO"
        message = "Following up on your earlier request."
    elif obs.stage == "GREETING":
        action_type = "ASK_QUESTION"
        message = "Great to connect. What are you mainly looking for?"
    elif obs.stage == "DISCOVERY":
        if obs.step_count <= 1:
            action_type = "ASK_QUESTION"
            message = "Could you share your budget and timeline?"
        else:
            action_type = "PROVIDE_INFO"
            message = "Thanks, based on that I can suggest the best option."
    elif obs.stage == "QUALIFICATION":
        action_type = "PROVIDE_INFO"
        message = "Let me explain exactly what is included and how support works."
    elif obs.stage == "OBJECTION_HANDLING":
        if (
            not _discount_used
            and obs.sentiment > 0.25
            and "low_trust" not in obs.uncertainties
            and obs.step_count >= 4
        ):
            _discount_used = True
            return Action(
                action_type="OFFER_DISCOUNT",
                message="I can offer a small limited discount to help you decide.",
                discount_pct=5.0,
            )
        action_type = "PROVIDE_INFO"
        message = "Totally fair concern. Let me address it directly."
    elif obs.stage == "NEGOTIATION":
        if (
            not _discount_used
            and obs.sentiment > 0.20
            and "low_trust" not in obs.uncertainties
            and obs.step_count >= 4
        ):
            _discount_used = True
            return Action(
                action_type="OFFER_DISCOUNT",
                message="I can do a small one-time 5% discount today.",
                discount_pct=5.0,
            )
        action_type = "PROVIDE_INFO"
        message = "Let me share one more detail that reduces your risk."
    elif obs.stage in {"CLOSING", "POST_SALE"}:
        action_type = "PROVIDE_INFO"
        message = "Excellent choice. I can help you complete this quickly."
    else:
        action_type, message = _STAGE_FALLBACK.get(
            obs.stage, ("ASK_QUESTION", "How can I help?")
        )

    if action_type == last_action_type and action_type in _NO_REPEAT:
        action_type = _NO_REPEAT[action_type]
        if action_type == "PROVIDE_INFO":
            message = "Let me share another useful detail."
        elif action_type == "GIVE_PRICE":
            message = "To keep this simple, here is a clear quote."
        elif action_type == "ASK_QUESTION":
            message = "Before we proceed, what matters most to you?"

    return Action(action_type=action_type, message=message)


def _build_action(llm_output: Dict[str, Any], obs: Observation, last_action_type: str = "") -> Action:
    """
    Convert LLM JSON dict to validated Action.
    Applies safety overrides, then falls back to stage heuristic on error.
    """
    global _discount_used

    action_type = str(llm_output.get("action_type", "")).upper()
    message = str(llm_output.get("message", ""))
    discount = llm_output.get("discount_pct")

    if action_type == "DELAY_RESPONSE":
        action_type = "PROVIDE_INFO"
        message = "Here's some more information that might help you decide."

    if action_type == "OFFER_DISCOUNT" and _discount_used:
        action_type = "PROVIDE_INFO"
        message = "Let me share more details about why this is a great choice."

    if action_type == "OFFER_DISCOUNT" and "low_trust" in obs.uncertainties:
        action_type = "PROVIDE_INFO"
        message = "Let me explain exactly what you're getting and why it's worth it."

    if action_type == "GIVE_PRICE" and obs.stage in {"GREETING", "DISCOVERY", "OBJECTION_HANDLING"}:
        action_type = "PROVIDE_INFO"
        message = "Let me first explain value and fit for your needs."

    if obs.obligations.has_pending and action_type not in {"PROVIDE_INFO", "ASK_QUESTION", "ESCALATE"}:
        action_type = "PROVIDE_INFO"
        message = "Let me follow up on your earlier request."

    if action_type == last_action_type and action_type in _NO_REPEAT:
        action_type = _NO_REPEAT[action_type]
        if not message:
            message = "Let me take a different approach."

    try:
        if action_type not in VALID_ACTIONS:
            raise ValueError(f"Unknown action: {action_type}")

        if action_type == "OFFER_DISCOUNT":
            pct = float(discount) if discount is not None else 10.0
            pct = max(5.0, min(15.0, pct))
            _discount_used = True
            return Action(action_type="OFFER_DISCOUNT", message=message, discount_pct=pct)

        return Action(action_type=action_type, message=message)

    except Exception:
        return _fallback_action_by_state(obs, last_action_type=last_action_type)


def run_episode(client: Optional[OpenAI], task_id: str) -> None:
    """Run one complete episode for the given task and emit structured logs."""
    global _discount_used
    _discount_used = False

    rewards: List[float] = []
    steps_taken: int = 0
    success: bool = False
    final_outcome: str = "IN_PROGRESS"
    last_action_type: str = ""

    log_start(task=task_id, model=MODEL_NAME)

    try:
        env = make_env(task_id=task_id)
        env.seed(TASK_SEEDS.get(task_id, 42))
        obs = env.reset()

        while not env.state().episode_done:
            step = steps_taken + 1

            llm_output = _call_llm(client, obs, step)
            action = _build_action(llm_output, obs, last_action_type=last_action_type)

            error_str: Optional[str] = None
            executed_action = action
            try:
                obs, reward, done, info = env.step(action)
            except Exception as exc:
                error_str = str(exc).replace("\n", " ")[:120]
                print(f"env.step() error: {error_str}", file=sys.stderr, flush=True)
                safe_action = Action(action_type="ASK_QUESTION", message="Could you tell me more?")
                obs, reward, done, info = env.step(safe_action)
                executed_action = safe_action

            rewards.append(reward)
            steps_taken = step
            final_outcome = info.get("outcome", "IN_PROGRESS")
            last_action_type = executed_action.action_type

            log_step(
                step=step,
                action=executed_action.action_type,
                reward=reward,
                done=done,
                error=error_str,
            )

            if done:
                break

    except Exception as exc:
        print(f"Episode failed: {exc}", file=sys.stderr, flush=True)
        final_outcome = "IN_PROGRESS"

    success = final_outcome in SUCCESS_OUTCOMES
    log_end(success=success, steps=steps_taken, rewards=rewards)


def main() -> None:
    client: Optional[OpenAI] = None
    if LLM_API_KEY:
        client = OpenAI(base_url=API_BASE_URL, api_key=LLM_API_KEY)
    else:
        print(
            "No HF_TOKEN or OPENAI_API_KEY provided; using deterministic fallback policy.",
            file=sys.stderr,
            flush=True,
        )

    for task_id in TASKS:
        run_episode(client, task_id)


if __name__ == "__main__":
    main()
