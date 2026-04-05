"""
agents.py — Baseline agents for WhatsApp RL.

All agents receive a Pydantic Observation object and return a Pydantic Action object.
Three agents are provided:
  - random_agent    : uniformly random valid action
  - rule_agent      : hand-crafted heuristic policy
  - heuristic_agent : richer multi-stage heuristic, useful as a stronger baseline
"""

import random as _random

from models import Action, ACTIONS, Observation


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_action(action_type: str, message: str = "", discount_pct: float = None) -> Action:
    """
    Convenience wrapper so every agent constructs Action the same way.
    Handles the OFFER_DISCOUNT special case cleanly in one place.
    """
    if action_type == "OFFER_DISCOUNT":
        return Action(
            action_type="OFFER_DISCOUNT",
            message=message,
            discount_pct=discount_pct if discount_pct is not None else 10.0,
        )
    return Action(action_type=action_type, message=message)


# ── random agent ──────────────────────────────────────────────────────────────

def random_agent(obs: Observation) -> Action:
    """
    Uniformly random valid action.
    OFFER_DISCOUNT is always paired with a random discount_pct so it passes
    Action's model_validator.
    """
    action_type = _random.choice(ACTIONS)

    if action_type == "OFFER_DISCOUNT":
        discount_pct = round(_random.uniform(5.0, 30.0), 1)
        return _make_action("OFFER_DISCOUNT", discount_pct=discount_pct)

    return _make_action(action_type)


# ── rule agent ────────────────────────────────────────────────────────────────

def rule_agent(obs: Observation) -> Action:
    """
    Simple rule-based heuristic policy.

    Priority order:
      1. Pending obligations → follow up with PROVIDE_INFO
      2. Negative sentiment  → offer a 10% discount to recover goodwill
      3. Default             → ask a qualifying question
    """
    # obs is a Pydantic Observation — use attribute access, NOT .get()
    if obs.obligations.has_pending:
        return _make_action("PROVIDE_INFO", message="Following up on your request!")

    if obs.sentiment < 0:
        return _make_action("OFFER_DISCOUNT", discount_pct=10.0)

    return _make_action("ASK_QUESTION", message="What are you looking for?")


# ── heuristic agent ───────────────────────────────────────────────────────────

def heuristic_agent(obs: Observation) -> Action:
    """
    Richer stage-aware heuristic. Useful as a stronger baseline than rule_agent.

    Decision logic:
      GREETING / DISCOVERY     → ask a question to qualify the lead
      QUALIFICATION            → provide info to build trust
      OBJECTION_HANDLING       → offer a small discount to break resistance
      NEGOTIATION              → offer a moderate discount to close
      CLOSING / POST_SALE      → provide info to reinforce the decision
      High annoyance (>0.7)    → escalate before losing the customer
      Pending obligations      → always service them first
      Default                  → ask a question
    """
    # service pending obligations first regardless of stage
    if obs.obligations.has_pending:
        return _make_action("PROVIDE_INFO", message="Just following up as promised!")

    # escalate if customer is very annoyed
    if obs.sentiment < -0.6:
        return _make_action("ESCALATE", message="Let me get someone senior to help.")

    stage = obs.stage

    if stage in ("GREETING", "DISCOVERY"):
        return _make_action("ASK_QUESTION", message="Could you tell me more about what you need?")

    if stage == "QUALIFICATION":
        return _make_action("PROVIDE_INFO", message="Here's what makes our product a great fit.")

    if stage == "OBJECTION_HANDLING":
        return _make_action("OFFER_DISCOUNT", discount_pct=10.0)

    if stage == "NEGOTIATION":
        return _make_action("OFFER_DISCOUNT", discount_pct=20.0)

    if stage in ("CLOSING", "POST_SALE"):
        return _make_action("PROVIDE_INFO", message="You've made a great choice. Here's what comes next.")

    # fallback
    return _make_action("ASK_QUESTION", message="How can I help you today?")