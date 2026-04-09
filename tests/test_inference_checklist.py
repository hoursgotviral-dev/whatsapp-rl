import io
import unittest
from contextlib import redirect_stdout

import inference


class InferenceChecklistTests(unittest.TestCase):
    def test_env_var_defaults_match_checklist(self):
        self.assertEqual(inference.API_BASE_URL, inference.os.getenv("API_BASE_URL", "https://router.huggingface.co/v1"))
        self.assertEqual(inference.MODEL_NAME, inference.os.getenv("MODEL_NAME", "Qwen/Qwen2.5-72B-Instruct"))
        self.assertIsNone(inference.HF_TOKEN)
        self.assertIsNone(inference.OPENAI_API_KEY)

    def test_log_start_matches_required_format(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            inference.log_start("task1", "demo-model")
        self.assertEqual(buf.getvalue().strip(), "[START] task=task1 env=whatsapp_sales_rl model=demo-model")

    def test_log_step_matches_required_format(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            inference.log_step(step=2, action="ASK_QUESTION", reward=0.12, done=False, error=None)
        self.assertEqual(
            buf.getvalue().strip(),
            "[STEP] step=2 action=ASK_QUESTION reward=0.12 done=false error=null",
        )

    def test_log_end_matches_required_format(self):
        buf = io.StringIO()
        with redirect_stdout(buf):
            inference.log_end(success=True, steps=3, rewards=[0.1, -0.2, 1.25])
        self.assertEqual(buf.getvalue().strip(), "[END] success=true steps=3 rewards=0.10,-0.20,1.25")

    def test_llm_api_key_falls_back_to_either_supported_env_var(self):
        self.assertEqual(inference.LLM_API_KEY, inference.HF_TOKEN or inference.OPENAI_API_KEY)


if __name__ == "__main__":
    unittest.main()
