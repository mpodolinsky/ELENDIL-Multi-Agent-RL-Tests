"""Evaluation script for trained AgileRL MADDPG agents with video recording."""

import os
import sys
import yaml
import wandb
import numpy as np
from pathlib import Path

# Add ELENDIL package to Python path
elendil_path = "/mnt/data/Documents/Project_M/ELENDIL"
if elendil_path not in sys.path:
    sys.path.insert(0, elendil_path)

from elendil.envs.grid_world_multi_agent import GridWorldEnvParallel

# Import the wrapper module directly
import importlib.util
spec = importlib.util.spec_from_file_location(
    "observation_flatten_wrapper", 
    os.path.join(os.path.dirname(__file__), '../wrappers/observation_flatten_wrapper.py')
)
observation_flatten_wrapper = importlib.util.module_from_spec(spec)
spec.loader.exec_module(observation_flatten_wrapper)
ObservationFlattenWrapper = observation_flatten_wrapper.ObservationFlattenWrapper

# AgileRL imports
from agilerl.algorithms.maddpg import MADDPG
import torch

def evaluate_model(checkpoint_path, model_id, num_episodes=5, record_video=True):
    """Evaluate a trained MADDPG model and optionally record videos."""
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cpu'
    print(f"Using device: {device}")
    
    # Load configuration files
    with open("configs/env_configs/medium_env_obstacles.yaml", "r") as f:
        env_config = yaml.safe_load(f)
    
    # Load agent configs
    agent_configs = []
    with open("configs/agent_configs/ground_agent.yaml", "r") as f:
        ground_config = yaml.safe_load(f)
        agent_configs.append(ground_config)
    
    with open("configs/agent_configs/air_observer_agent.yaml", "r") as f:
        air_config = yaml.safe_load(f)
        agent_configs.append(air_config)
    
    with open("configs/target_configs/target_config.yaml", "r") as f:
        target_config = yaml.safe_load(f)
    
    # Create evaluation environment with rendering
    print("🏗️  Creating evaluation environment...")
    base_env = GridWorldEnvParallel(
        size=env_config["size"],
        max_steps=env_config["max_steps"],
        render_mode="rgb_array" if record_video else None,
        show_fov_display=env_config["show_fov_display"],
        intrinsic=env_config.get("intrinsic", False),
        lambda_fov=env_config["lambda_fov"],
        show_target_coords=env_config.get("show_target_coords", False),
        no_target=env_config["no_target"],
        agents=agent_configs,
        target_config=target_config,
        enable_obstacles=env_config["enable_obstacles"],
        num_obstacles=env_config["num_obstacles"],
        num_visual_obstacles=env_config["num_visual_obstacles"],
    )
    
    wrapped_env = ObservationFlattenWrapper(base_env)
    
    print(f"✅ Environment created")
    print(f"✅ Agents: {wrapped_env.agents}")
    print(f"✅ Observation spaces: {wrapped_env.observation_space(wrapped_env.agents[0])}")
    print(f"✅ Action space: {wrapped_env.action_space(wrapped_env.agents[0])}")
    
    # Get observation and action spaces
    observation_spaces = [wrapped_env.observation_space(agent) for agent in wrapped_env.agents]
    action_spaces = [wrapped_env.action_space(agent) for agent in wrapped_env.agents]
    agent_ids = wrapped_env.agents
    
    # Load the trained agent
    print(f"📂 Loading checkpoint from {checkpoint_path}...")
    agent = MADDPG(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        device=device,
    )
    
    agent.load_checkpoint(checkpoint_path)
    print("✅ Model loaded successfully")
    
    # Extract model name from checkpoint path for video directory naming
    model_name = os.path.basename(checkpoint_path).replace('.pt', '')
    
    # Create video directory with model_id and model name
    if record_video:
        video_dir = Path(f"wandb/latest-run/videos/evaluation_{model_id}")
        video_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate for multiple episodes
    episode_rewards = {agent_id: [] for agent_id in agent_ids}
    
    for episode in range(num_episodes):
        obs, info = wrapped_env.reset()
        episode_running_rewards = {agent_id: 0.0 for agent_id in agent_ids}
        frames = []
        done = False
        step_count = 0
        
        while not done and step_count < env_config["max_steps"]:
            # Get actions
            actions, _ = agent.get_action(obs=obs, infos=info)
            
            # Render and save frame if recording
            if record_video:
                frame = base_env.render()
                if frame is not None:
                    frames.append(frame)
            
            # Step environment
            next_obs, rewards, terminations, truncations, infos = wrapped_env.step(actions)
            
            # Accumulate rewards
            for agent_id in agent_ids:
                episode_running_rewards[agent_id] += rewards[agent_id]
            
            obs = next_obs
            step_count += 1
            
            # Check if done
            done = all(terminations.values()) or all(truncations.values())
        
        # Record episode results
        for agent_id in agent_ids:
            episode_rewards[agent_id].append(episode_running_rewards[agent_id])
        
        print(f"Episode {episode + 1}/{num_episodes}: {episode_running_rewards}")
        
        # Save video if recording
        if record_video and frames:
            import imageio
            video_path = video_dir / f"episode_{episode + 1}.mp4"
            imageio.mimsave(str(video_path), frames, fps=10)
            print(f"📹 Saved video to {video_path}")
    
    # Upload videos directly to wandb run if recording
    if record_video and video_dir.exists():
        print(f"\n📤 Uploading videos to wandb...")
        try:
            # Check for latest-run and connect to it if no active run
            if wandb.run is None:
                latest_run_symlink = Path("wandb/latest-run")
                if latest_run_symlink.exists() and latest_run_symlink.is_symlink():
                    # Read the actual target directory
                    target_path = latest_run_symlink.resolve()
                    # Extract run ID from directory name (format: run-TIMESTAMP-ID)
                    run_dir_name = target_path.name
                    parts = run_dir_name.split('-')
                    if len(parts) >= 3:
                        run_id = parts[-1]  # Get the last part (the unique ID)
                        print(f"📋 Found latest run: {run_id}")
                        
                        # Try to resume the run
                        try:
                            # Use the same project name as the training scripts
                            project_name = "ELENDIL"
                            
                            # Initialize wandb to resume the existing run
                            wandb.init(id=run_id, resume="allow", project=project_name)
                            print(f"✅ Connected to run: {run_id}")
                        except Exception as e:
                            print(f"⚠️  Could not resume run {run_id}: {e}")
                            print("🔄 Initializing new wandb run instead...")
                            wandb.init(project=project_name)
                    else:
                        print("⚠️  Could not parse run ID from directory name")
                        print("🔄 Initializing new wandb run...")
                        wandb.init(project="ELENDIL")
                else:
                    print("⚠️  No latest-run found. Starting new wandb session...")
                    wandb.init(project="ELENDIL")
            
            if wandb.run is not None:
                # Log each video directly to the wandb run
                for video_file in sorted(video_dir.glob("*.mp4")):
                    video_name = video_file.stem  # e.g., "episode_1"
                    wandb.log({f"evaluation/videos/{video_name}": wandb.Video(str(video_file))})
                    print(f"  ✅ Uploaded {video_file.name} to wandb")
                print(f"✅ Successfully uploaded {num_episodes} videos to wandb")
            else:
                print("⚠️  Could not connect to wandb run. Videos saved locally only.")
        except Exception as e:
            print(f"⚠️  Error uploading videos to wandb: {e}")
            import traceback
            traceback.print_exc()
    
    # Print average rewards
    print("\n📊 Average rewards over episodes:")
    for agent_id in agent_ids:
        avg_reward = np.mean(episode_rewards[agent_id])
        print(f"  {agent_id}: {avg_reward:.2f}")
    
    wrapped_env.close()


if __name__ == "__main__":
    # import argparse

    import argparse

    parser = argparse.ArgumentParser(description="Evaluate trained MADDPG model")
    parser.add_argument("model_id", help="Model ID string, e.g., _0_10000")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to evaluate")
    parser.add_argument("--no_video", action="store_true", help="Don't record videos")
    args = parser.parse_args()

    checkpoint_path = f"wandb/latest-run/{args.model_id}.pt"
    evaluate_model(
        checkpoint_path=checkpoint_path,
        model_id=args.model_id,
        num_episodes=args.num_episodes,
        record_video=not args.no_video
    )

    # Example usage:
    # python agilerl_maddpg_eval.py _0_10000 --num_episodes 5 --no_video