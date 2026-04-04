from typing import Dict, Any, Tuple
from models import Observation, Action  # NEW IMPORT

def compute_step_reward(
    obs: Observation,  # CHANGED: was state_before/after
    action: Action,    # SIMPLIFIED
    info: Dict[str, Any],  # CHANGED: was user_event + done
    state_before: Dict[str, Any] = None,  # BACKWARD COMPAT
    state_after: Dict[str, Any] = None,   # BACKWARD COMPAT  
) -> Tuple[float, Dict[str, float]]:
    """
    Compute total reward and components for one step.
    
    obs: Current Observation (Pydantic)
    action: Action object (Pydantic)
    info: Contains user_event, done, state snapshots (backwards compat)
    """
    # Backward compatibility
    user_event = info.get("user_event", "reply")
    done = info.get("done", False)
    state_before = state_before or info.get("state_before", {})
    state_after = state_after or info.get("state_after", {})
    
    components: Dict[str, float] = {
        "engagement": 0.0,
        "obligations": 0.0,
        "relevance": 0.0,
        "spam": 0.0,
        "cost": 0.0,
        "final_outcome": 0.0,
    }

    # --- Engagement: reward user replies + stage progress ---
    if user_event != "no_reply":
        if info.get("stage_after") != info.get("stage_before"):  # From info
            components["engagement"] += 0.05

    # --- Relevance: use Observation fields ---
    components["relevance"] = 0.1 if action.action_type == "PROVIDE_INFO" else 0.0

    # --- Spam / annoyance penalty ---
    annoyance_after = state_after.get("annoyance", obs.sentiment * -0.5)  # Fallback to obs
    if action.action_type not in ("WAIT",) and annoyance_after > 0.7 and user_event == "no_reply":
        components["spam"] -= 0.05

    # --- Cost penalty ---
    cost_before = state_before.get("cost_to_business", 0.0)
    cost_after = state_after.get("cost_to_business", 0.0)
    delta_cost = max(0.0, cost_after - cost_before)
    if delta_cost > 0:
        components["cost"] -= delta_cost

    total_reward = sum(components.values())

    # --- Final outcome reward on terminal step ---
    if done:
        outcome = state_after.get("outcome", "unresolved")
        final_satisfaction = state_after.get("satisfaction", obs.sentiment)
        final_annoyance = state_after.get("annoyance", 0.0)
        total_cost = state_after.get("cost_to_business", 0.0)

        outcome_bonus = 0.0
        if outcome == "converted":
            outcome_bonus = 3.0 * final_satisfaction - 1.5 * final_annoyance - 0.5 * total_cost
        elif outcome == "churned":
            outcome_bonus = -2.0 - final_annoyance
        elif outcome == "escalated":
            outcome_bonus = 1.0 * final_satisfaction - 0.5 * total_cost
        else:  # unresolved
            outcome_bonus = 0.5 * final_satisfaction - final_annoyance

        components["final_outcome"] = outcome_bonus
        total_reward += outcome_bonus

    return float(total_reward), components

# TEST FUNCTION (remove after testing)
if __name__ == "__main__":
    from models import Observation, Action
    obs = Observation(
        chat_history=[], stage="start", intent="inquiry", 
        sentiment=0.5, step_count=1
    )
    action = Action(action_type="PROVIDE_INFO")
    info = {"user_event": "reply", "done": False}
    
    reward, components = compute_step_reward(obs, action, info)
    print(f"Test reward: {reward:.3f}")
    print(f"Components: {components}")