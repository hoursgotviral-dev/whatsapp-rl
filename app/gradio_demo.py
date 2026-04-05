import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import gradio as gr

try:
    from env import make_env
    ENV_OK = True
    print("ENV + Pydantic OK!")
except Exception as e:
    ENV_OK = False
    print(f"Import failed: {e}")


# ── action inference from user message ────────────────────────────────────────

def _infer_action(message: str) -> dict:
    """
    Map the user's free-text message to one of the 7 env action types.
    This is a simple heuristic so the demo exercises all action types.
    """
    msg = message.lower().strip()

    if any(w in msg for w in ["discount", "cheaper", "deal", "offer", "reduce"]):
        # try to parse a number, default to 10%
        import re
        nums = re.findall(r'\d+', msg)
        pct = float(nums[0]) if nums else 10.0
        pct = min(pct, 50.0)  # cap at 50%
        return {"action_type": "OFFER_DISCOUNT", "discount_pct": pct}

    if any(w in msg for w in ["price", "cost", "how much", "rate", "fee"]):
        return {"action_type": "GIVE_PRICE", "message": message}

    if any(w in msg for w in ["?", "what", "which", "who", "where", "when", "why", "how"]):
        return {"action_type": "ASK_QUESTION", "message": message}

    if any(w in msg for w in ["info", "detail", "tell me", "explain", "feature", "spec"]):
        return {"action_type": "PROVIDE_INFO", "message": message}

    if any(w in msg for w in ["manager", "escalate", "supervisor", "complaint"]):
        return {"action_type": "ESCALATE", "message": message}

    if any(w in msg for w in ["bye", "goodbye", "end", "stop", "quit", "done"]):
        return {"action_type": "END_CONVERSATION", "message": message}

    if any(w in msg for w in ["wait", "later", "hold on", "give me a moment"]):
        return {"action_type": "DELAY_RESPONSE", "message": message}

    # default
    return {"action_type": "PROVIDE_INFO", "message": message}


# ── core chat function ─────────────────────────────────────────────────────────

def safe_chat(task_id: str, message: str, history: list, env_state):
    """
    Called on every Send click. env_state persists across calls via gr.State.
    """
    if not ENV_OK:
        return history, "❌ Environment not ready", env_state

    if not message.strip():
        return history, "Type a message first.", env_state

    try:
        env = env_state  # retrieve the persistent env

        # guard: if env is None or episode is done, start fresh
        if env is None or env.state().episode_done:
            clean_task = task_id.split("-")[0]
            env = make_env(clean_task)
            env.reset()

        action = _infer_action(message)
        from models import Action
        if action["action_type"] == "OFFER_DISCOUNT":
            action_obj = Action(action_type="OFFER_DISCOUNT", discount_pct=action.get("discount_pct", 10.0), message=action.get("message", ""))
        else:
            action_obj = Action(action_type=action["action_type"], message=action.get("message", ""))
        obs, reward, done, info = env.step(action_obj)

        # build the agent's reply from chat history
        agent_reply = obs.chat_history[-1] if obs.chat_history else "..."

        history = history + [
    {"role": "user", "content": message},
    {"role": "assistant", "content": agent_reply}
]

        status_lines = [
            f"Stage: {obs.stage}  |  Step: {obs.step_count}  |  Action: {action_obj.action_type}",
            f"Reward: {reward:+.3f}  |  Sentiment: {obs.sentiment:+.2f}  |  Done: {done}",
        ]
        if done:
            status_lines.append(f"🏁 Episode ended — outcome: {info['outcome']}")

        metrics = "\n".join(status_lines)
        return history, metrics, env  # ← return updated env back into gr.State

    except Exception as e:
        import traceback
        tb = traceback.format_exc()[-300:]
        return history, f"Error: {e}\n{tb}", env_state


def reset_env(task_id: str):
    """Called by the New Episode button. Returns fresh env + cleared UI."""
    if not ENV_OK:
        return [], "❌ Environment not ready", None

    try:
        clean_task = task_id.split("-")[0]
        env = make_env(clean_task)
        obs = env.reset()
        s = env.state()

        metrics = (
            f"New episode started — task: {clean_task}\n"
            f"Stage: {obs.stage}  |  User type: {s.user_type}  |  "
            f"Trust: {s.trust:.2f}  |  Patience: {s.patience:.2f}"
        )
        return [], metrics, env  # clear history, store new env in gr.State

    except Exception as e:
        return [], f"Reset error: {e}", None


# ── UI ─────────────────────────────────────────────────────────────────────────

print("WhatsApp RL Demo starting...")
print("Open: http://localhost:7860")

with gr.Blocks(title="WhatsApp RL Agent") as demo:
    gr.Markdown("# WhatsApp Business RL Demo")
    gr.Markdown(
        "Each message is mapped to one of 7 agent actions and stepped through the RL env. "
        "The episode persists across messages. Click **New Episode** to reset."
    )

    # persistent env lives here — survives across Send clicks
    env_state = gr.State(value=None)

    with gr.Row():
        task_dropdown = gr.Dropdown(
            ["task1-easy", "task2-medium", "task3-hard"],
            value="task1-easy",
            label="Customer Difficulty",
            scale=2,
        )
        new_episode_btn = gr.Button("New Episode", variant="secondary", scale=1)

    chatbot = gr.Chatbot(height=420, label="Conversation")
    msg = gr.Textbox(
        placeholder="e.g. 'What's the price?' / 'Can I get a discount?' / 'Tell me more'",
        label="Your message (agent action is inferred from keywords)",
        lines=1,
    )

    with gr.Row():
        send_btn = gr.Button("Send", variant="primary")
        clear_btn = gr.Button("Clear chat only")

    metrics = gr.Textbox(label="Episode metrics", lines=3, interactive=False)

    # Send: passes env_state in, gets updated env_state back
    send_btn.click(
        safe_chat,
        inputs=[task_dropdown, msg, chatbot, env_state],
        outputs=[chatbot, metrics, env_state],
    )
    msg.submit(
        safe_chat,
        inputs=[task_dropdown, msg, chatbot, env_state],
        outputs=[chatbot, metrics, env_state],
    )

    # New Episode: resets env and clears chat
    new_episode_btn.click(
        reset_env,
        inputs=[task_dropdown],
        outputs=[chatbot, metrics, env_state],
    )

    # Clear: wipes chat UI only, does NOT touch env_state
    clear_btn.click(lambda: ([], ""), outputs=[chatbot, metrics])

demo.launch(server_name="0.0.0.0", server_port=7861, share=False, show_error=True)