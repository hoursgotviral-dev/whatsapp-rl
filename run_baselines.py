from env import make_env
from agents.agents import random_agent, rule_agent, heuristic_agent
from reward.grading import grade_trajectory

agents = {
    "random_agent":    random_agent,
    "rule_agent":      rule_agent,
    "heuristic_agent": heuristic_agent,
}

tasks = [
    ("task1", 42),
    ("task2", 43),
    ("task3", 44),
]

results = {}

for agent_name, agent_fn in agents.items():
    scores = []
    for task_id, seed in tasks:
        env = make_env(task_id)
        env.seed(seed)
        obs = env.reset()
        trajectory = []
        done = False
        while not done:
            action = agent_fn(obs)
            obs, reward, done, info = env.step(action)
            trajectory.append((obs, action, reward, info))
        score = grade_trajectory(trajectory, task_id=task_id)
        scores.append(score)
        print(f"{agent_name} | {task_id} | score={score:.4f} | outcome={info['outcome']}")
    mean = sum(scores) / len(scores)
    results[agent_name] = scores + [mean]
    print()

print("\n--- FINAL TABLE ---")
print(f"{'Agent':<20} {'Task1':>8} {'Task2':>8} {'Task3':>8} {'Mean':>8}")
for agent_name, scores in results.items():
    print(f"{agent_name:<20} {scores[0]:>8.4f} {scores[1]:>8.4f} {scores[2]:>8.4f} {scores[3]:>8.4f}")
