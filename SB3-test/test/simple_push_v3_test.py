# evaluate_ippo_hemac.py
"""
Run a trained IPPO model on the HeMAC environment with human rendering.
This script loads the latest checkpoint and runs 10 evaluation episodes.
"""

import os
import torch
import numpy as np
from gymnasium import spaces
from mpe2 import simple_push_v3, simple_adversary_v3
from agilerl.algorithms import IPPO

# -----------------------
# Configuration
# -----------------------
CHECKPOINT_PATH = "wandb/latest-run/checkpoints/ippo_step860160.pt"  # <-- adjust as needed
NUM_EPISODES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Environment Setup
# -----------------------
env = simple_adversary_v3.parallel_env(
    render_mode="human")
env.reset()

observation_spaces = [env.observation_space(agent) for agent in env.agents]
action_spaces = [env.action_space(agent) for agent in env.agents]
agent_ids = list(env.agents)

# -----------------------
# Load Agent
# -----------------------
agent = IPPO(
    observation_spaces=observation_spaces,
    action_spaces=action_spaces,
    agent_ids=agent_ids,
    device=DEVICE,
)
agent.load_checkpoint(CHECKPOINT_PATH)
print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

# agent.test(env) #TODO There is a testing function directly within agileRL


# -----------------------
# Evaluation Loop
# -----------------------
for ep in range(NUM_EPISODES):
    obs, info = env.reset()
    terminated = {a_id: False for a_id in agent_ids}
    truncated = {a_id: False for a_id in agent_ids}
    total_reward = 0.0

    print(f"\n--- Episode {ep+1}/{NUM_EPISODES} ---")

    while not all(terminated.values()) and not all(truncated.values()):
        with torch.no_grad():
            action, _, _, _ = agent.get_action(obs=obs, infos=info)

        # Normalize, clip, and format actions
        formatted_action = {}
        for a_id, act in action.items():
            a_space = env.action_space(a_id)

            if isinstance(a_space, spaces.Discrete):
                # Flatten and cast to int (e.g. [[0]] → 0)
                act = int(np.squeeze(act))

            elif isinstance(a_space, spaces.Box):
                # Flatten, clip, and cast to float32
                act = np.squeeze(act).astype(np.float32)
                act = np.clip(act, a_space.low, a_space.high)

            formatted_action[a_id] = act

        # Step environment safely
        try:
            obs, reward, termination, truncation, info = env.step(formatted_action)
        except AssertionError as e:
            print("\n--- ACTION OUT OF BOUNDS ---")
            for a_id, act in formatted_action.items():
                a_space = env.action_space(a_id)
                if not a_space.contains(act):
                    print(f"Agent '{a_id}' invalid action: {act}")
                    print(f"Expected space: {a_space}")
            raise e

        total_reward += sum(reward.values())
        terminated = termination
        truncated = truncation

    print(f"Episode {ep+1} total reward: {total_reward:.2f}")

env.close()
print("\nEvaluation complete.")
