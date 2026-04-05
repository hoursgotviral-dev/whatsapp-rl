"""
Baseline agents for WhatsApp RL
"""

def random_agent(obs):
    """Random action agent"""
    import random
    actions = ["PROVIDE_INFO", "ASK_QUESTION", "OFFER_DISCOUNT", "GIVE_PRICE"]
    return {"action_type": random.choice(actions)}

def rule_agent(obs):
    """Simple rule-based agent"""
    if obs.get("obligations", {}).get("has_pending", False):
        return {"action_type": "PROVIDE_INFO", "message": "Following up!"}
    if obs.get("sentiment", 0) < 0:
        return {"action_type": "OFFER_DISCOUNT", "discount_pct": 10.0}
    return {"action_type": "ASK_QUESTION", "message": "What are you looking for?"}