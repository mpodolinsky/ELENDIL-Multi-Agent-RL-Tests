"""
evaluate_ippo_hemac_manual_record.py

Runs a trained IPPO agent in HeMAC, captures frames manually, and writes video files.
"""
import glob
import os
import torch
import numpy as np
import imageio
from gymnasium import spaces
from hemac import HeMAC_v0
from agilerl.algorithms import IPPO
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.utils.hemac_utils import make_parallel_env




# def create_env():
#     """Initialize the HeMAC environment with RGB rendering enabled."""
#     env = HeMAC_v0.parallel_env(
#         render_mode="rgb_array",
#         n_drones=1,
#         n_observers=1,
#         n_provisioners=0,
#         rescuing_targets=False,
#         area_size=(1000, 1000),
#         max_cycles=300,
#         min_obstacles=0,
#         max_obstacles=0,
#         poi_config=[{"speed": 1.0, "dimension": [8, 8], "spawn_mode": "random"}],
#     )
#     return env


def format_action(env, agent_id, act):
    a_space = env.action_space(agent_id)
    if isinstance(a_space, spaces.Discrete):
        act = int(np.squeeze(act))
    elif isinstance(a_space, spaces.Box):
        act = np.squeeze(act).astype(np.float32)
        act = np.clip(act, a_space.low, a_space.high)
    return act

# Function find the last file in a directory
def find_last_file(directory: str, pattern: str = "*.pt") -> str | None:
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        return None
    return max(files, key=os.path.getctime)


def record_tests(hemac_config: str, CHECKPOINT_PATH: str = None, NUM_EPISODES: int = 10, VIDEO_FOLDER: str = 'wandb/latest-run/videos'):
    FPS = 30  # frames per second for output video
    os.makedirs(VIDEO_FOLDER, exist_ok=True)

    CHECKPOINT_PATH = find_last_file("wandb/latest-run/checkpoints") if CHECKPOINT_PATH is None else CHECKPOINT_PATH

    env = make_parallel_env(hemac_config=hemac_config, render_mode="rgb_array")

    # Prepare agent
    agent_ids = list(env.possible_agents)
    obs_spaces = [env.observation_space(a) for a in agent_ids]
    act_spaces = [env.action_space(a) for a in agent_ids]
    
    agent = IPPO(
        observation_spaces=obs_spaces,
        action_spaces=act_spaces,
        agent_ids=agent_ids,
        # device= torch.device("cpu"),  # will move to device later
    )
    agent.load_checkpoint(CHECKPOINT_PATH)
    # Force everything to CPU
    agent.device = "cpu"
    agent.actors.to("cpu")
    agent.critics.to("cpu")

    # Optional: if your algorithm has target networks
    if hasattr(agent, "target_actor"):
        agent.target_actor.to("cpu")
    if hasattr(agent, "target_critic"):
        agent.target_critic.to("cpu")

    print(f"Loaded checkpoint from {CHECKPOINT_PATH}")

    for ep in range(NUM_EPISODES):
        obs, info = env.reset()
        terminated = {a: False for a in agent_ids}
        truncated = {a: False for a in agent_ids}

        frames = []
        print(f"\n--- Episode {ep + 1}/{NUM_EPISODES} ---")

        # collect first frame
        frame = env.render()
        frames.append(frame)

        total_reward = 0.0
        while not (all(terminated.values()) or all(truncated.values())):

            with torch.no_grad():
                action, _, _, _ = agent.get_action(obs=obs, infos=info)

            # format per-agent actions
            formatted = {}
            for a_id, act in action.items():
                formatted[a_id] = format_action(env, a_id, act)

            obs, reward, termination, truncation, info = env.step(formatted)
            terminated = termination
            truncated = truncation
            total_reward += sum(reward.values())

            frame = env.render()
            frames.append(frame)

        print(f"Episode {ep + 1} total reward: {total_reward:.2f}") 

        # save the frames as a video file (e.g., mp4 or gif)
        video_path = os.path.join(VIDEO_FOLDER, f"episode_{ep+1:03d}.mp4")
        imageio.mimsave(video_path, frames, fps=FPS)
        print(f"Saved video: {video_path}")

    env.close()
    print("Recording complete.")


if __name__ == "__main__":
        # -----------------------
    # Configuration
    # -----------------------
    # Current best : "wandb/run-20251016_005621-98cij82h/checkpoints/ippo_step1000000.pt"
    # CHECKPOINT_PATH = "wandb/latest-run/checkpoints/ippo_step778240.pt"  # <-- adjust as needed
    # USING unidirectional sensor 1000*1000 CHECKPOINT_PATH = "wandb/latest-run/checkpoints/ippo_step778240.pt"  # <-- adjust as needed
    # CHECKPOINT_PATH = "wandb/run-20251016_143504-55rv17yv/checkpoints/ippo_step737280.pt"  # <-- adjust as needed
    CHECKPOINT_PATH = "wandb/latest-run/checkpoints/ippo_step880640.pt"
    NUM_EPISODES = 3
    VIDEO_FOLDER = "videos/hemac_test"

    record_tests(hemac_config="hemac_env_config.yaml", NUM_EPISODES=NUM_EPISODES, VIDEO_FOLDER="wandb/latest-run/training-videos")
