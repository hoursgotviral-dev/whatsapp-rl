"""
inference.py — WhatsApp Sales RL Inference Script
==================================================
Runs one episode per task (task1, task2, task3) using an LLM agent.

Required environment variables:
  API_BASE_URL   – OpenAI-compatible endpoint (e.g. https://router.huggingface.co/v1)
  MODEL_NAME     – Model identifier (e.g. Qwen/Qwen2.5-72B-Instruct)
  HF_TOKEN       – API key / Hugging Face token

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
from typing import List, Optional

from openai import OpenAI

# ── make project importable from root ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from env import make_env
from models import Action, Observation

# ── env vars ──────────────────────────────────────────────────────────────────
API_BASE_URL = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME   = os.getenv("MODEL_NAME",   "Qwen/Qwen2.5-72B-Instruct")
API_KEY      = os.getenv("HF_TOKEN") or os.getenv("API_KEY", "")

BENCHMARK    = "whatsapp_sales_rl"
TASKS        = ["task1", "task2", "task3"]

TEMPERATURE  = 0.3
MAX_TOKENS   = 256

# SALE or ESCALATED are both non-negative terminal outcomes
SUCCESS_OUTCOMES = {"SALE", "ESCALATED"}


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING HELPERS
# ══════════════════════════════════════════════════════════════════════════════

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


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = textwrap.dedent("""
You are a WhatsApp sales agent. Your goal is to raise the customer's conversion
probability to close a SALE by building trust and satisfaction — NOT by spamming discounts.

CRITICAL RULES:
1. OFFER_DISCOUNT is expensive — use it AT MOST ONCE per episode, only when sentiment > 0.2
   and the customer has explicitly resisted the price. Keep discount_pct between 5 and 15.
2. Never repeat the same action twice in a row.
3. DELAY_RESPONSE is forbidden — never use it.
4. Build trust first: ASK_QUESTION → PROVIDE_INFO → GIVE_PRICE → small OFFER_DISCOUNT → close.
5. If sentiment is already positive (> 0.1), skip discounts and use PROVIDE_INFO to reinforce.
6. If uncertainties include "low_trust", use PROVIDE_INFO or ASK_QUESTION, not discounts.
7. If uncertainties include "low_patience", wrap up quickly — use GIVE_PRICE or PROVIDE_INFO.

Ideal action sequence by stage:
  GREETING          → ASK_QUESTION
  DISCOVERY         → ASK_QUESTION once, then PROVIDE_INFO
  QUALIFICATION     → PROVIDE_INFO or GIVE_PRICE
  OBJECTION_HANDLING→ PROVIDE_INFO first; OFFER_DISCOUNT (5-10%) only if still resistant
  NEGOTIATION       → OFFER_DISCOUNT once (10-15% max), then PROVIDE_INFO
  CLOSING           → PROVIDE_INFO to reinforce decision
  POST_SALE         → PROVIDE_INFO

Reply with a valid JSON object and nothing else. No extra text.
  {"action_type": "<ACTION>", "message": "<your message to the customer>"}

For OFFER_DISCOUNT:
  {"action_type": "OFFER_DISCOUNT", "discount_pct": 10, "message": "<msg>"}

Available actions:
  ASK_QUESTION, GIVE_PRICE, OFFER_DISCOUNT, PROVIDE_INFO,
  ESCALATE, DELAY_RESPONSE, END_CONVERSATION
""").strip()


def _build_user_prompt(obs: Observation, step: int) -> str:
    history_text = "\n".join(obs.chat_history[-6:]) if obs.chat_history else "None"
    uncertainties = ", ".join(obs.uncertainties) if obs.uncertainties else "none"
    pending = obs.obligations.pending
    obligations_text = (
        "\n".join(f"  - [{o.type}] {o.description}" for o in pending)
        if pending else "  none"
    )

    # Coaching hint driven by live state signals
    if "low_patience" in obs.uncertainties:
        hint = "WARNING: Customer patience is low. Wrap up fast — use GIVE_PRICE or PROVIDE_INFO."
    elif "low_trust" in obs.uncertainties:
        hint = "WARNING: Trust is low. Do NOT offer discounts. Use PROVIDE_INFO or ASK_QUESTION."
    elif "high_annoyance" in obs.uncertainties:
        hint = "WARNING: Customer is annoyed. Be concise — use PROVIDE_INFO, not ASK_QUESTION."
    elif step <= 2:
        hint = "Early stage: ask one focused question to understand needs."
    elif step <= 5:
        hint = "Mid stage: provide value. Share product info or quote a price."
    elif obs.sentiment > 0.1:
        hint = "Sentiment is positive. Reinforce with PROVIDE_INFO and move toward closing."
    else:
        hint = "Late stage: offer a small one-time discount (5-10%) to break resistance."

    return textwrap.dedent(f"""
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
    """).strip()


# ══════════════════════════════════════════════════════════════════════════════
# LLM CALL
# ══════════════════════════════════════════════════════════════════════════════

def _call_llm(client: OpenAI, obs: Observation, step: int) -> dict:
    """
    Ask the LLM for an action. Returns a dict with at minimum 'action_type'.
    Falls back to an empty dict (triggers heuristic) on any error.
    """
    user_prompt = _build_user_prompt(obs, step)
    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        raw = (completion.choices[0].message.content or "").strip()

        # Strip markdown fences if the model wraps its JSON
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
            raw = raw.strip()

        parsed = json.loads(raw)
        return parsed

    except Exception as exc:
        print(f"[DEBUG] LLM call failed at step {step}: {exc}", flush=True)
        return {}


# ══════════════════════════════════════════════════════════════════════════════
# ACTION BUILDER
# ══════════════════════════════════════════════════════════════════════════════

VALID_ACTIONS = {
    "ASK_QUESTION", "GIVE_PRICE", "OFFER_DISCOUNT",
    "PROVIDE_INFO", "ESCALATE", "DELAY_RESPONSE", "END_CONVERSATION",
}

# Stage-aware fallback policy (used when LLM output is invalid/missing)
_STAGE_FALLBACK = {
    "GREETING":           ("ASK_QUESTION",   "How can I help you today?"),
    "DISCOVERY":          ("ASK_QUESTION",   "Could you tell me more about what you're looking for?"),
    "QUALIFICATION":      ("PROVIDE_INFO",   "Here's what makes our product a great fit for you."),
    "OBJECTION_HANDLING": ("PROVIDE_INFO",   "Let me address your concerns directly."),
    "NEGOTIATION":        ("OFFER_DISCOUNT", "10"),
    "CLOSING":            ("PROVIDE_INFO",   "You've made a great choice!"),
    "POST_SALE":          ("PROVIDE_INFO",   "Thank you for your purchase!"),
    "ESCALATED":          ("ESCALATE",       "Let me connect you with someone senior."),
    "ENDED":              ("END_CONVERSATION","Thank you for your time."),
}

# Module-level flag: only one discount per episode
_discount_used: bool = False


def _build_action(llm_output: dict, obs: Observation) -> Action:
    """
    Convert LLM JSON dict → validated Action.
    Applies safety overrides, then falls back to stage heuristic on error.
    """
    global _discount_used

    action_type = str(llm_output.get("action_type", "")).upper()
    message     = str(llm_output.get("message", ""))
    discount    = llm_output.get("discount_pct")

    # ── safety overrides ──────────────────────────────────────────────────────

    # Block DELAY_RESPONSE always
    if action_type == "DELAY_RESPONSE":
        action_type = "PROVIDE_INFO"
        message = "Here's some more information that might help you decide."

    # Block repeated discounts — only one per episode
    if action_type == "OFFER_DISCOUNT" and _discount_used:
        action_type = "PROVIDE_INFO"
        message = "Let me share more details about why this is a great choice."

    # Block discounts when trust is low — they don't help and cost a lot
    if action_type == "OFFER_DISCOUNT" and "low_trust" in obs.uncertainties:
        action_type = "PROVIDE_INFO"
        message = "Let me explain exactly what you're getting and why it's worth it."

    # Fulfil pending obligations immediately
    if obs.obligations.has_pending and action_type not in {"PROVIDE_INFO", "ASK_QUESTION", "ESCALATE"}:
        action_type = "PROVIDE_INFO"
        message = "Let me follow up on your earlier request."

    # ── build Action ──────────────────────────────────────────────────────────
    try:
        if action_type not in VALID_ACTIONS:
            raise ValueError(f"Unknown action: {action_type}")

        if action_type == "OFFER_DISCOUNT":
            pct = float(discount) if discount is not None else 10.0
            pct = max(5.0, min(15.0, pct))   # hard cap: 5–15%
            _discount_used = True
            return Action(action_type="OFFER_DISCOUNT", message=message, discount_pct=pct)

        return Action(action_type=action_type, message=message)

    except Exception:
        # Stage-aware fallback
        fallback_type, fallback_msg = _STAGE_FALLBACK.get(
            obs.stage, ("ASK_QUESTION", "How can I help?")
        )
        if fallback_type == "OFFER_DISCOUNT" and not _discount_used:
            _discount_used = True
            return Action(
                action_type="OFFER_DISCOUNT",
                message="Here's a special discount for you.",
                discount_pct=10.0,
            )
        elif fallback_type == "OFFER_DISCOUNT":
            return Action(action_type="PROVIDE_INFO", message="Let me share more details.")
        return Action(action_type=fallback_type, message=fallback_msg)


# ══════════════════════════════════════════════════════════════════════════════
# EPISODE RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_episode(client: OpenAI, task_id: str) -> None:
    """Run one complete episode for the given task and emit structured logs."""
    global _discount_used
    _discount_used = False          # reset per episode

    rewards:       List[float] = []
    steps_taken:   int         = 0
    success:       bool        = False
    final_outcome: str         = "IN_PROGRESS"

    log_start(task=task_id, model=MODEL_NAME)

    try:
        env = make_env(task_id=task_id)
        obs = env.reset()

        while not env.state().episode_done:
            step = steps_taken + 1

            # Ask LLM for action
            llm_output = _call_llm(client, obs, step)
            action     = _build_action(llm_output, obs)

            error_str: Optional[str] = None
            try:
                obs, reward, done, info = env.step(action)
            except Exception as exc:
                error_str = str(exc).replace("\n", " ")[:120]
                print(f"[DEBUG] env.step() error: {error_str}", flush=True)
                # Guaranteed-safe fallback
                safe_action = Action(
                    action_type="ASK_QUESTION",
                    message="Could you tell me more?",
                )
                obs, reward, done, info = env.step(safe_action)

            rewards.append(reward)
            steps_taken   = step
            final_outcome = info.get("outcome", "IN_PROGRESS")

            log_step(
                step=step,
                action=action.action_type,
                reward=reward,
                done=done,
                error=error_str,
            )

            if done:
                break

    except Exception as exc:
        print(f"[DEBUG] Episode failed: {exc}", flush=True)
        final_outcome = "IN_PROGRESS"

    success = final_outcome in SUCCESS_OUTCOMES
    log_end(success=success, steps=steps_taken, rewards=rewards)


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    client = OpenAI(base_url=API_BASE_URL, api_key=API_KEY)
    for task_id in TASKS:
        run_episode(client, task_id)


if __name__ == "__main__":
    main()