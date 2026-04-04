import random
from models import Action, ACTIONS, Observation


class BaseAgent:
    def act(self, observation: Observation) -> Action:
        raise NotImplementedError


class RandomAgent(BaseAgent):
    def act(self, observation: Observation) -> Action:
        return Action(action_type=random.choice(ACTIONS))


class RuleAgent(BaseAgent):
    def act(self, observation: Observation) -> Action:

        if observation.stage == "CLOSING":
            return Action(action_type="END_CONVERSATION")

        if observation.stage == "NEGOTIATION":
            return Action(action_type="OFFER_DISCOUNT", discount_pct=10.0)

        if observation.stage == "DISCOVERY":
            return Action(action_type="ASK_QUESTION")

        if observation.stage == "GREETING":
            return Action(action_type="ASK_QUESTION")

        return Action(action_type="PROVIDE_INFO")


class HeuristicAgent(BaseAgent):
    def act(self, observation: Observation) -> Action:

        if observation.sentiment < -0.3:
            return Action(action_type="PROVIDE_INFO")

        if "low user trust" in observation.uncertainties:
            return Action(action_type="ASK_QUESTION")

        if "patience critically low" in observation.uncertainties:
            return Action(action_type="GIVE_PRICE")

        if observation.stage == "NEGOTIATION":
            return Action(action_type="OFFER_DISCOUNT", discount_pct=5.0)

        if observation.stage == "CLOSING":
            return Action(action_type="END_CONVERSATION")

        if observation.stage == "DISCOVERY":
            return Action(action_type="ASK_QUESTION")

        if observation.stage == "GREETING":
            return Action(action_type="PROVIDE_INFO")

        return Action(action_type="PROVIDE_INFO")