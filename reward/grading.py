"""
Trajectory grading for OpenEnv.
Input: list of (obs, action, reward, info) tuples
Output: single float score (0-10 range)
"""

from typing import List, Tuple, Any
from dataclasses import dataclass

@dataclass
class TrajectoryScore:
    total_reward: float
    success_rate: float
    final_score: float
    completion_rate: float

def grade_trajectory(trajectory: List[Tuple[Any, Any, float, Any]]) -> float:
    """
    Grades complete episode trajectory.
    
    Args:
        trajectory: List of (obs, action, step_reward, info) tuples
        
    Returns:
        Final score (0-10 scale)
    """
    if not trajectory:
        return 0.0
    
    # Basic scoring
    step_rewards = [r for _, _, r, _ in trajectory]
    total_reward = sum(step_rewards)
    avg_reward = total_reward / len(trajectory)
    
    # Success rate (steps with positive reward)
    success_steps = sum(1 for r in step_rewards if r > 0)
    success_rate = success_steps / len(trajectory)
    
    # Completion bonus (longer episodes = harder)
    completion_rate = min(len(trajectory) / 20.0, 1.0)
    
    # Combined score (0-10 scale)
    final_score = (avg_reward * 3 + success_rate * 4 + completion_rate * 3) * 10
    
    return min(final_score, 10.0)  # Cap at 10

# Test function
if __name__ == "__main__":
    fake_trajectory = [({}, {}, 0.3, {})] * 10
    score = grade_trajectory(fake_trajectory)
    print(f"Test score: {score:.2f}")