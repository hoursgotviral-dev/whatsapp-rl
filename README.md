---
title: Whatsapp RL
emoji: 🤖
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---
# WhatsApp Business Conversation RL Environment

An OpenEnv-compliant reinforcement learning environment that simulates real-world WhatsApp B2C sales conversations. An agent learns to optimise business outcomes — conversion rate, customer satisfaction, cost management — through sequential decision-making across multi-turn dialogues.

---

## Why this environment?

WhatsApp is the primary B2C sales channel for hundreds of millions of businesses globally. Every conversation is a sequential decision problem: the agent must choose when to ask questions, when to present price, when to offer a discount, and when to escalate — all while managing a customer whose trust, patience, and annoyance are shifting in real time. Getting it wrong means a lost sale or a damaged relationship. Getting it right requires exactly the kind of long-horizon reasoning that RL agents are built for.

This environment models that problem faithfully: stochastic customer personas, a 9-stage conversation funnel, a trackable obligation system (promises made and broken), and a shaped reward that provides signal on every step — not just at the moment of conversion.

---

## Environment overview

| Property | Value |
|---|---|
| Interface | OpenEnv v1.0 (`reset` / `step` / `state`) |
| Action space | Discrete — 7 action types |
| Observation space | Structured Pydantic object (see below) |
| Episode length | 15 steps (easy) · 20 steps (medium) · 25 steps (hard) |
| Termination conditions | Sale, abandonment, step limit, escalation, or explicit end |
| Reward | Dense shaped reward + terminal bonus |
| Stochasticity | Customer persona sampled at reset; user responses stochastic |

---

## Action space

The agent selects one action per step from 7 discrete types.

| Action type | Required fields | Effect on hidden state |
|---|---|---|
| `ASK_QUESTION` | `message` | +0.03 trust, +0.02 satisfaction |
| `GIVE_PRICE` | `message` | +0.03 conversion probability |
| `OFFER_DISCOUNT` | `message`, `discount_pct` (0–100) | +0.10 conversion prob + 0.002 × pct, +0.05 satisfaction, cost increases by pct |
| `PROVIDE_INFO` | `message` | +0.05 trust, +0.03 satisfaction |
| `ESCALATE` | `message` | −0.10 annoyance (relief), +0.05 trust; terminates episode as ESCALATED |
| `DELAY_RESPONSE` | `message` | −0.10 trust, +0.20 annoyance, −0.10 conversion prob, −0.30 step reward |
| `END_CONVERSATION` | `message` | Terminates episode immediately as NO_SALE |

`OFFER_DISCOUNT` requires `discount_pct` to be set. All other actions must not include it. This is enforced at construction time by Pydantic's model validator.

**Constructing actions (Python):**

```python
from models import Action

action = Action(action_type="ASK_QUESTION", message="What is your budget?")
action = Action(action_type="OFFER_DISCOUNT", message="Here's a deal", discount_pct=15.0)
```

---

## Observation space

Each call to `reset()` and `step()` returns an `Observation` Pydantic object with the following fields.

| Field | Type | Description |
|---|---|---|
| `chat_history` | `List[str]` | Full conversation so far, alternating `AGENT:` and `USER:` prefixes |
| `stage` | `str` | Current conversation stage (see stage progression below) |
| `intent` | `str` | Inferred user intent: `PURCHASE` or `INQUIRY` |
| `sentiment` | `float` | Noisy estimate of user sentiment in [−1.0, +1.0]. Derived from true satisfaction with Gaussian noise (σ = 0.05) |
| `uncertainties` | `List[str]` | Active warning flags: `low_trust`, `low_patience`, `high_annoyance`, `N_obligation_violations` |
| `obligations` | `ObligationSummary` | Pending, fulfilled, and violated commitments (see obligations below) |
| `step_count` | `int` | Number of steps taken in this episode |

**Accessing observations (Python):**

```python
obs = env.reset()

print(obs.stage)               # "GREETING"
print(obs.sentiment)           # e.g. 0.12
print(obs.obligations.has_pending)       # True / False
print(obs.obligations.violation_count)   # int
print(obs.uncertainties)       # e.g. ["low_trust"]
```

### Conversation stages

The episode progresses through up to 9 stages driven by user responses and agent actions. Stage transitions are deterministic given the current stage and the user's sentiment event.

```
GREETING → DISCOVERY → QUALIFICATION → NEGOTIATION → CLOSING → POST_SALE
                    ↘                ↘             ↗
                  OBJECTION_HANDLING ──────────────
                                              ↓ (ESCALATE action)
                                          ESCALATED
                                              ↓ (END_CONVERSATION action)
                                           ENDED
```

Advancing through later stages (QUALIFICATION, NEGOTIATION, CLOSING) provides conversion probability boosts of 0.05, 0.08, and 0.12 per step respectively.

### Obligation system

When a user says something like "remind me tomorrow" or "I'll get back to you", the environment creates a `follow_up` obligation. When the agent's message contains a commitment phrase ("I'll send", "I'll check", "I'll follow up"), an `agent_commitment` obligation is created. Each obligation has a deadline (default: 4 steps from creation). If the agent does not service a pending obligation before its deadline, it is marked `EXPIRED`, and the environment applies:

- +0.15 annoyance
- −0.10 trust
- −0.10 satisfaction

These are multiplied by the obligation's importance weight (0–1).

---

## Hidden state (ground truth)

The full ground-truth `State` is not visible to the agent but is returned by `env.state()` and included as `state_snapshot` in the `info` dict of each `step()` call. It is intended for debugging, monitoring, and grader evaluation.

| Field | Type | Range | Description |
|---|---|---|---|
| `user_type` | `str` | — | `IMPULSIVE`, `ANALYTICAL`, `SKEPTICAL`, `LOYAL`, or `PRICE_SENSITIVE` |
| `true_intent` | `str` | — | True underlying intent (always `PURCHASE` in current tasks) |
| `trust` | `float` | [0, 1] | How much the customer trusts the agent |
| `patience` | `float` | [0, 1] | Decays by 0.05 per step; episode ends if it hits 0.15 |
| `annoyance` | `float` | [0, 1] | Accumulated friction; raised by delays and obligation violations |
| `satisfaction` | `float` | [0, 1] | Drives the noisy sentiment signal in observations |
| `conversion_prob` | `float` | [0, 1] | Episode ends as SALE when this reaches 0.85 |
| `cost_to_business` | `float` | [0, ∞) | Cumulative discount cost; penalises the reward |
| `stage` | `str` | — | Current conversation stage |
| `outcome` | `str` | — | `IN_PROGRESS`, `SALE`, `NO_SALE`, `ABANDONED`, or `ESCALATED` |

---

## Reward function

The reward is dense — every step produces a non-zero signal.

**Per-step components:**

| Component | Formula |
|---|---|
| Satisfaction gain | `satisfaction_after − satisfaction_before` |
| Annoyance penalty | `−(annoyance_after − annoyance_before)` |
| Obligation penalty | `−0.2 × violation_count` |
| Delay penalty | `−0.30` if action is `DELAY_RESPONSE`, else 0 |

**Terminal bonus (on the final step only):**

| Outcome | Bonus |
|---|---|
| `SALE` | `+2.0 − 0.01 × cost_to_business` |
| `ABANDONED` | `−1.5` |
| `NO_SALE` | `−0.5` |
| `ESCALATED` | `0.0` |

The shape of the reward encourages the agent to increase satisfaction steadily, avoid annoyance spikes, service obligations before their deadlines, never delay without reason, and close with a conversion rather than abandonment.

---

## Tasks

Three tasks with increasing difficulty. Each is defined by the initial customer state distribution, persona weights, episode length, and reward scaling.

### Task 1 — Easy Sale

A warm, receptive customer who is already interested in the product. The agent must confirm needs, present value clearly, and close efficiently.

| Parameter | Value |
|---|---|
| `max_steps` | 15 |
| `trust` initial | [0.50, 0.70] |
| `patience` initial | [0.75, 0.95] |
| `conversion_prob` initial | [0.40, 0.60] |
| Dominant persona | IMPULSIVE (35%), LOYAL (20%) |
| Terminal conversion bonus | ×2.0 |

**Grader success criteria:** episode ends as `SALE` with fewer than 3 obligation violations.

### Task 2 — Balanced Negotiation

A moderate customer with mixed signals — shows interest but raises objections and price concerns. The agent must qualify needs, handle objections skillfully, and negotiate to close.

| Parameter | Value |
|---|---|
| `max_steps` | 20 |
| `trust` initial | [0.40, 0.60] |
| `patience` initial | [0.60, 0.90] |
| `conversion_prob` initial | [0.30, 0.50] |
| Dominant persona | PRICE_SENSITIVE (25%), ANALYTICAL (25%) |
| Terminal conversion bonus | ×2.0 |

**Grader success criteria:** episode ends as `SALE` with final `satisfaction ≥ 0.6` and `annoyance ≤ 0.4`.

### Task 3 — Hard Close

A skeptical, analytical customer who is hard to convince and quick to abandon. The agent must build trust patiently, resolve deep objections, and avoid any missteps that raise annoyance.

| Parameter | Value |
|---|---|
| `max_steps` | 25 |
| `trust` initial | [0.20, 0.45] |
| `patience` initial | [0.40, 0.65] |
| `conversion_prob` initial | [0.15, 0.35] |
| Dominant persona | SKEPTICAL (35%), ANALYTICAL (30%) |
| Terminal conversion bonus | ×2.5 |

**Grader success criteria:** episode ends as `SALE` before step 20 with `trust ≥ 0.6` at termination.

---

## Baseline scores

Three agents were evaluated over 100 episodes per task. All runs seeded with `seed=42` for reproducibility.

| Agent | Task 1 score | Task 2 score | Task 3 score | Mean |
|---|---|---|---|---|
| `random_agent` | 0.21 | 0.14 | 0.08 | 0.14 |
| `rule_agent` | 0.43 | 0.31 | 0.17 | 0.30 |
| `heuristic_agent` | 0.61 | 0.48 | 0.29 | 0.46 |

Scores are grader outputs in [0.0, 1.0]. A frontier LLM agent with access to the full observation is expected to score 0.70+ on Task 1 and 0.45+ on Task 3.

> **Note:** update this table with your actual `inference.py` run before submission. The numbers above are placeholders from the heuristic baseline run.

---

## Project structure

```
.
├── env/
│   ├── __init__.py          # make_env() factory
│   ├── environment.py       # WhatsAppEnv: reset / step / state
│   └── simulator/
│       └── user_simulator.py  # UserSimulator: stochastic customer responses
├── models.py                # Pydantic models: Action, Observation, State, ObligationSummary
├── reward/
│   ├── __init__.py
│   ├── core.py              # compute_step_reward()
│   └── grading.py           # grade_trajectory() — scores 0.0–1.0
├── tasks/
│   ├── __init__.py
│   └── configs.py           # TaskConfig definitions for task1 / task2 / task3
├── agents/
│   ├── __init__.py
│   └── agents.py            # random_agent, rule_agent, heuristic_agent
├── ui/
│   └── gradio_demo.py       # interactive Gradio demo (stateful multi-turn)
├── server.py                # FastAPI OpenEnv server
├── inference.py             # baseline inference script (OpenAI client)
├── openenv.yaml             # OpenEnv task metadata
├── requirements.txt         # Python dependencies
├── Dockerfile               # container definition
└── README.md                # this file
```

---

## Setup

### Local (Python)

```bash
# clone the repo
git clone https://github.com/<your-org>/whatsapp-rl.git
cd whatsapp-rl

# install dependencies
pip install -r requirements.txt

# start the OpenEnv server
uvicorn server:app --host 0.0.0.0 --port 7860 --reload

# in a separate terminal — run the Gradio demo
python ui/gradio_demo.py

# run the baseline inference script
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4o-mini"
export HF_TOKEN="your_token_here"
python inference.py
```

### Docker

```bash
# build the image
docker build -t whatsapp-rl .

# run the server
docker run -p 7860:7860 \
  -e API_BASE_URL="https://api.openai.com/v1" \
  -e MODEL_NAME="gpt-4o-mini" \
  -e HF_TOKEN="your_token_here" \
  whatsapp-rl

# health check
curl http://localhost:7860/health
```

### Environment variables

| Variable | Required | Description |
|---|---|---|
| `API_BASE_URL` | Yes (inference) | LLM API base URL, e.g. `https://api.openai.com/v1` |
| `MODEL_NAME` | Yes (inference) | Model identifier, e.g. `gpt-4o-mini` |
| `HF_TOKEN` | Yes (inference) | Hugging Face or OpenAI API key |

---

## API reference

The FastAPI server exposes three endpoints matching the OpenEnv spec.

### `POST /v1/reset?task_id=task1`

Start a new episode. `task_id` must be `task1`, `task2`, or `task3`.

Returns the initial `Observation` as JSON.

```bash
curl -X POST "http://localhost:7860/v1/reset?task_id=task1"
```

### `POST /v1/step`

Advance one step. Body must be a valid action JSON.

```bash
curl -X POST "http://localhost:7860/v1/step" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "ASK_QUESTION", "message": "What is your budget?"}'

curl -X POST "http://localhost:7860/v1/step" \
  -H "Content-Type: application/json" \
  -d '{"action_type": "OFFER_DISCOUNT", "discount_pct": 15.0}'
```

Returns `StepResponse` containing observation, reward, done flag, outcome, reward components, and obligation events.

### `GET /v1/state`

Returns the full ground-truth `State` (hidden from agents in real training). Useful for debugging.

```bash
curl http://localhost:7860/state
```

### `GET /health`

Liveness probe. Returns server version, uptime, current task, and episode status.

```bash
curl http://localhost:7860/health
```

---

## Using the environment directly in Python

```python
from env import make_env
from models import Action

# create and reset
env = make_env(task_id="task1")
obs = env.reset()

print(obs.stage)        # "GREETING"
print(obs.sentiment)    # e.g. 0.08

# run an episode
done = False
total_reward = 0.0

while not done:
    # your agent logic here
    action = Action(action_type="ASK_QUESTION", message="Tell me about your needs.")
    obs, reward, done, info = env.step(action)
    total_reward += reward

print(f"Outcome: {info['outcome']}  |  Total reward: {total_reward:.3f}")

# inspect ground truth (not available to agent during training)
state = env.state()
print(f"Final trust: {state.trust:.2f}  |  Conversion prob: {state.conversion_prob:.2f}")
```

---

## Running the baseline agents

```python
from env import make_env
from agents.agents import random_agent, rule_agent, heuristic_agent
from reward.grading import grade_trajectory

env = make_env("task2")
obs = env.reset()

trajectory = []
done = False

while not done:
    action = heuristic_agent(obs)
    obs, reward, done, info = env.step(action)
    trajectory.append((obs, action, reward, info))

score = grade_trajectory(trajectory)
print(f"Score: {score:.3f}")   # 0.0 – 1.0
```

---

## Requirements

```
pydantic>=2.5.0
fastapi>=0.100.0
uvicorn[standard]>=0.25.0
numpy>=1.24.0
requests>=2.31.0
openai>=1.0.0
gradio>=4.0.0
python-multipart
```

Python 3.10 or higher required.

---

## Evaluation methodology

Each task is graded by `grade_trajectory()` in `reward/grading.py`, which produces a score in [0.0, 1.0]. The grader is deterministic — given the same trajectory, it always returns the same score. It considers:

- Whether the episode ended as a `SALE` (primary signal)
- Final satisfaction and annoyance levels
- Number of obligation violations
- Step efficiency (earlier conversion = higher score on hard tasks)

Grader scores are distinct from cumulative episode reward — a high reward is possible without a sale (e.g. by keeping satisfaction high for many steps), but a high grader score requires actual conversion.