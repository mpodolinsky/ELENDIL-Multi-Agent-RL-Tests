"""
Test script to verify the parallel environment works correctly.
"""
import sys
import os

# Add HA-SPO2V-Env to path FIRST before any gymnasium_env imports
ha_spo2v_path = "/mnt/data/Documents/Project_M/HA-SPO2V-Env"
if ha_spo2v_path not in sys.path:
    sys.path.insert(0, ha_spo2v_path)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import yaml
from envs.parallel_gridworld import parallel_env

# Load configurations
with open("configs/env_configs/small_env_obstacles.yaml", "r") as f:
    env_config = yaml.safe_load(f)

with open("configs/agent_configs/ground_agent.yaml", "r") as f:
    agent_config = yaml.safe_load(f)

with open("configs/target_configs/target_config.yaml", "r") as f:
    target_config = yaml.safe_load(f)

print("Creating parallel environment...")

# Create parallel environment
env = parallel_env(
    size=env_config["size"],
    max_steps=env_config["max_steps"],
    render_mode=None,  # Set to None for testing
    show_fov_display=env_config["show_fov_display"],
    intrinsic=env_config.get("intrinsic", False),
    lambda_fov=env_config["lambda_fov"],
    show_target_coords=env_config.get("show_target_coords", False),
    no_target=env_config["no_target"],
    agents=[agent_config],
    target_config=target_config,
    enable_obstacles=env_config["enable_obstacles"],
    num_obstacles=env_config["num_obstacles"],
    num_visual_obstacles=env_config["num_visual_obstacles"],
)

print("\n" + "="*60)
print("Environment created successfully!")
print("="*60)

print(f"\nAgents: {env.agents}")
print(f"Observation spaces: {env.observation_spaces}")
print(f"Action spaces: {env.action_spaces}")

# Test reset
print("\n" + "="*60)
print("Testing reset...")
print("="*60)

observations, infos = env.reset()
print(f"Observations keys: {observations.keys()}")
print(f"Infos keys: {infos.keys()}")

# Test step with random actions
print("\n" + "="*60)
print("Testing step with random actions...")
print("="*60)

actions = {agent: env.action_space(agent).sample() for agent in env.agents}
print(f"Actions: {actions}")

observations, rewards, terminations, truncations, infos = env.step(actions)

print(f"\nRewards: {rewards}")
print(f"Terminations: {terminations}")
print(f"Truncations: {truncations}")

# Run a few more steps
print("\n" + "="*60)
print("Running 5 more steps...")
print("="*60)

for i in range(5):
    actions = {agent: env.action_space(agent).sample() for agent in env.agents}
    observations, rewards, terminations, truncations, infos = env.step(actions)
    print(f"Step {i+1}: Rewards={rewards}, Done={any(terminations.values()) or any(truncations.values())}")
    
    if all(terminations.values()) or all(truncations.values()):
        print("Episode finished!")
        break

env.close()

print("\n" + "="*60)
print("✅ Parallel environment test passed!")
print("="*60)

