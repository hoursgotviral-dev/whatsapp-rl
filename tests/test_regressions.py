import unittest
from unittest.mock import patch

import gradio as gr

from env import make_env
from models import Action
from reward.grading import grade_trajectory

# Prevent side effects when importing app.gradio_demo in tests.
gr.Blocks.launch = lambda self, *args, **kwargs: None

from app.gradio_demo import _build_action, _fallback_action


class EnvironmentRegressionTests(unittest.TestCase):
    def test_positive_user_event_applies_additive_delta(self):
        env = make_env("task2")
        env.reset()
        env._state = env.state().with_updates(
            satisfaction=0.42,
            conversion_prob=0.35,
            trust=0.51,
            annoyance=0.0,
        )

        with patch.object(env, "_simulator", return_value=("Good to know!", "positive")):
            env.step(Action(action_type="PROVIDE_INFO", message="Here are the details."))

        state = env.state()
        self.assertAlmostEqual(state.satisfaction, 0.55, places=6)
        self.assertGreater(state.conversion_prob, 0.35)

    def test_skeptical_user_event_reduces_values_without_zeroing(self):
        env = make_env("task2")
        env.reset()
        env._state = env.state().with_updates(
            trust=0.60,
            conversion_prob=0.50,
            satisfaction=0.55,
            annoyance=0.0,
        )

        with patch.object(env, "_simulator", return_value=("I'll need to verify that.", "skeptical")):
            env.step(Action(action_type="PROVIDE_INFO", message="Here are the details."))

        state = env.state()
        self.assertGreater(state.trust, 0.0)
        self.assertGreater(state.conversion_prob, 0.0)
        self.assertAlmostEqual(state.annoyance, 0.05, places=6)

    def test_escalation_is_not_a_success_outcome_for_grading(self):
        escalated = grade_trajectory([
            ({}, {}, 0.0, {
                "outcome": "ESCALATED",
                "state_snapshot": {"satisfaction": 0.7, "annoyance": 0.1},
                "violation_count": 0,
            })
        ])
        sale = grade_trajectory([
            ({}, {}, 0.0, {
                "outcome": "SALE",
                "state_snapshot": {"satisfaction": 0.7, "annoyance": 0.1},
                "violation_count": 0,
            })
        ])
        self.assertLess(escalated, sale)

    def test_reward_weights_are_applied(self):
        env = make_env("task2")
        env.config.reward_weights = {"stage_progress": 0.0}
        env.reset()
        env._state = env.state().with_updates(
            satisfaction=0.50,
            trust=0.55,
            conversion_prob=0.40,
        )

        with patch.object(env, "_simulator", return_value=("Thanks, that's helpful.", "neutral")):
            _, reward, _, info = env.step(Action(action_type="ASK_QUESTION", message="What are you looking for?"))

        self.assertEqual(info["reward_components"]["stage_progress"], 0.0)
        self.assertAlmostEqual(reward, 0.02, places=6)

    def test_task_specific_grader_penalizes_task3_slow_low_trust_sale(self):
        weak_task3_sale = grade_trajectory([
            ({}, {}, 0.0, {
                "outcome": "SALE",
                "time_step": 22,
                "state_snapshot": {
                    "satisfaction": 0.8,
                    "annoyance": 0.1,
                    "trust": 0.45,
                    "time_step": 22,
                },
                "violation_count": 0,
            })
        ], task_id="task3")
        strong_task3_sale = grade_trajectory([
            ({}, {}, 0.0, {
                "outcome": "SALE",
                "time_step": 12,
                "state_snapshot": {
                    "satisfaction": 0.8,
                    "annoyance": 0.1,
                    "trust": 0.75,
                    "time_step": 12,
                },
                "violation_count": 0,
            })
        ], task_id="task3")
        self.assertLess(weak_task3_sale, strong_task3_sale)

    def test_post_sale_terminates_as_sale(self):
        env = make_env("task2")
        env.reset()
        env._state = env.state().with_updates(stage="POST_SALE")
        done = env._check_done()
        self.assertTrue(done)
        self.assertTrue(env.state().episode_done)
        self.assertEqual(env.state().outcome, "SALE")

    def test_demo_fallback_does_not_repeat_same_action(self):
        env = make_env("task1")
        obs = env.reset()
        action, source = _fallback_action(obs, "ASK_QUESTION", "llm_unavailable")
        self.assertNotEqual(action.action_type, "ASK_QUESTION")
        self.assertIn("fallback", source)

    def test_demo_build_action_changes_repeated_llm_action(self):
        env = make_env("task1")
        obs = env.reset()
        action, source = _build_action(
            {"action_type": "PROVIDE_INFO", "message": "Let me share more details."},
            obs,
            last_action_type="PROVIDE_INFO",
        )
        self.assertNotEqual(action.action_type, "PROVIDE_INFO")
        self.assertEqual(source, "adjusted:no_repeat")


if __name__ == "__main__":
    unittest.main()
