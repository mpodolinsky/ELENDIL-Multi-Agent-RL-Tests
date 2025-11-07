"""Evaluation script for trained AgileRL MADDPG agents with video recording."""

import os
import sys
import yaml
import wandb
import numpy as np
from pathlib import Path
from gymnasium.spaces import Discrete

# Add ELENDIL package to Python path
elendil_path = "/mnt/data/Documents/Project_M/ELENDIL"
if elendil_path not in sys.path:
    sys.path.insert(0, elendil_path)

from elendil.envs.grid_world_multi_agent import GridWorldEnvParallel, GridWorldEnvParallelExploration

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

def evaluate_model(checkpoint_path, model_id, num_episodes=5, record_video=True, run_dir=None, project_name=None):
    """Evaluate a trained MADDPG model and optionally record videos.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_id: Model ID string
        num_episodes: Number of episodes to evaluate
        record_video: Whether to record videos
        run_dir: Original run directory path (for video saving location)
        project_name: WandB project name
    """
    
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = 'cuda'
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
    base_env = GridWorldEnvParallelExploration(
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
    print(f"✅ Agent IDs in loaded model: {agent.agent_ids}")
    print(f"✅ Environment agent IDs: {agent_ids}")
    
    # Ensure agent IDs match
    if set(agent.agent_ids) != set(agent_ids):
        print(f"⚠️  Warning: Agent IDs mismatch! Using agent IDs from loaded model: {agent.agent_ids}")
        agent_ids = agent.agent_ids
    
    # Extract model name from checkpoint path for video directory naming
    model_name = os.path.basename(checkpoint_path).replace('.pt', '')
    
    # Create video directory with model_id and model name
    if record_video:
        # Use run_dir if provided, otherwise use latest-run
        base_dir = run_dir if run_dir else "wandb/latest-run"
        video_dir = Path(base_dir) / "videos" / f"evaluation_{model_id}"
        video_dir.mkdir(parents=True, exist_ok=True)
    
    # Evaluate for multiple episodes
    episode_rewards = {agent_id: [] for agent_id in agent_ids}
    
    for episode in range(num_episodes):
        obs, info = wrapped_env.reset()
        episode_running_rewards = {agent_id: 0.0 for agent_id in agent_ids}
        frames = []
        done = False
        step_count = 0
        # Store last observation for each agent (for terminated agents)
        last_obs = {agent_id: obs.get(agent_id, None) for agent_id in agent_ids}
        
        while not done and step_count < env_config["max_steps"]:
            # Get actions
            # The agent expects ALL agent_ids to be present in observations
            # If an agent is terminated and not in obs, we need to provide a default observation
            filtered_obs = {}
            filtered_info = {}
            for agent_id in agent.agent_ids:
                if agent_id in obs:
                    filtered_obs[agent_id] = obs[agent_id]
                    filtered_info[agent_id] = info.get(agent_id, {})
                else:
                    # Agent is terminated - use last known observation
                    if last_obs[agent_id] is not None:
                        filtered_obs[agent_id] = last_obs[agent_id]
                    else:
                        # Fallback: use zero observation
                        obs_space = wrapped_env.observation_space(agent_id)
                        if hasattr(obs_space, 'shape'):
                            filtered_obs[agent_id] = np.zeros(obs_space.shape, dtype=obs_space.dtype)
                        else:
                            filtered_obs[agent_id] = np.zeros(27, dtype=np.float32)
                    filtered_info[agent_id] = {}
            
            if not filtered_obs:
                print(f"⚠️  No valid observations for agent.agent_ids: {agent.agent_ids}")
                print(f"   Available obs keys: {list(obs.keys())}")
                break
            
            try:
                actions, _ = agent.get_action(obs=filtered_obs, infos=filtered_info)
            except KeyError as e:
                print(f"❌ Error in get_action: {e}")
                print(f"   filtered_obs keys: {list(filtered_obs.keys())}")
                print(f"   agent.agent_ids: {agent.agent_ids}")
                print(f"   obs keys (from env): {list(obs.keys())}")
                raise
            
            # Ensure actions dict has all agents from obs (for terminated agents, use last action or no-op)
            full_actions = {}
            for agent_id in obs.keys():
                if agent_id in actions:
                    full_actions[agent_id] = actions[agent_id]
                else:
                    # Agent is terminated, use a default action (no-op)
                    action_space = wrapped_env.action_space(agent_id)
                    if isinstance(action_space, Discrete):
                        full_actions[agent_id] = 0  # No-op action
                    else:
                        full_actions[agent_id] = np.zeros(action_space.shape, dtype=action_space.dtype)
            actions = full_actions
            
            # Render and save frame if recording
            if record_video:
                frame = base_env.render()
                if frame is not None:
                    frames.append(frame)
            
            # Step environment
            next_obs, rewards, terminations, truncations, infos = wrapped_env.step(actions)
            
            # Update last observations for all agents
            for agent_id in agent_ids:
                if agent_id in next_obs:
                    last_obs[agent_id] = next_obs[agent_id]
            
            # Accumulate rewards (only for agents that are still active)
            # Note: In multi-agent environments, terminated agents may not appear in rewards dict
            for agent_id in agent_ids:
                if agent_id in rewards:
                    episode_running_rewards[agent_id] += rewards[agent_id]
                # If agent is not in rewards but was in agent_ids, it's likely terminated
                # The running reward stays at its last value (which is fine)
            
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
            video_filename = f"{model_id}_episode_{episode + 1}.mp4"
            video_path = video_dir / video_filename
            imageio.mimsave(str(video_path), frames, fps=10)
            print(f"📹 Saved video to {video_path}")
    
    # Log videos to wandb if recording
    if record_video and video_dir.exists():
        print(f"\n📤 Logging videos to wandb...")
        try:
            # Ensure wandb run is active
            if wandb.run is None:
                latest_run_symlink = Path("wandb/latest-run")
                if latest_run_symlink.exists() and latest_run_symlink.is_symlink():
                    target_path = latest_run_symlink.resolve()
                    run_dir_name = target_path.name
                    parts = run_dir_name.split('-')
                    if len(parts) >= 3:
                        run_id = parts[-1]
                        print(f"📋 Found latest run: {run_id}")
                        try:
                            # Use provided project_name, or try to determine it
                            if project_name is None:
                                project_name = "ELENDIL"  # Default fallback
                                try:
                                    api = wandb.Api()
                                    # Try to find the run by ID (may need to search across projects)
                                    # First, try reading from the same config file as training script
                                    config_path = Path("configs/train_configs/agilerl_maddpg_config.yaml")
                                    if config_path.exists():
                                        with open(config_path) as f:
                                            config = yaml.safe_load(f)
                                            project_name = config.get("wandb", {}).get("project", "ELENDIL")
                                    else:
                                        # Fallback: try to get project from wandb run metadata
                                        # Search in common project names
                                        for proj in ["ELENDIL", "ELENDIL-dummy"]:
                                            try:
                                                run = api.run(f"{proj}/{run_id}")
                                                project_name = proj
                                                break
                                            except:
                                                continue
                                except Exception as e:
                                    print(f"⚠️  Could not determine project name: {e}")
                                    # Fallback to reading config
                                    try:
                                        config_path = Path("configs/train_configs/agilerl_maddpg_config.yaml")
                                        if config_path.exists():
                                            with open(config_path) as f:
                                                config = yaml.safe_load(f)
                                                project_name = config.get("wandb", {}).get("project", "ELENDIL")
                                    except:
                                        pass
                            wandb.init(id=run_id, resume="allow", project=project_name)
                            print(f"✅ Connected to run: {run_id} in project: {project_name}")
                        except Exception as e:
                            print(f"⚠️  Could not resume run: {e}")
                            wandb.init(project=project_name)
                else:
                    # Fallback: try to read from config file or use provided project_name
                    if project_name is None:
                        project_name = "ELENDIL"
                        try:
                            config_path = Path("configs/train_configs/agilerl_maddpg_config.yaml")
                            if config_path.exists():
                                with open(config_path) as f:
                                    config = yaml.safe_load(f)
                                    project_name = config.get("wandb", {}).get("project", "ELENDIL")
                        except:
                            pass
                    wandb.init(project=project_name)
            
            # Log each video using wandb.Video
            if wandb.run is not None:
                for video_file in sorted(video_dir.glob(f"{model_id}_*.mp4")):
                    video_name = video_file.stem  # e.g., "_0_10000_episode_1"
                    wandb.log({f"evaluation/videos/{video_name}": wandb.Video(str(video_file))})
                    print(f"  ✅ Logged {video_file.name} to wandb")
                print(f"✅ Successfully logged videos for {model_id} to wandb")
            else:
                print("⚠️  Could not connect to wandb run. Videos saved locally only.")
        except Exception as e:
            print(f"⚠️  Error logging videos to wandb: {e}")
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
    parser.add_argument("--run_dir", type=str, default=None, help="Original run directory path (before resume)")
    parser.add_argument("--project_name", type=str, default=None, help="WandB project name")
    args = parser.parse_args()

    # Use provided run_dir, or fallback to latest-run
    if args.run_dir and os.path.exists(args.run_dir):
        checkpoint_path = os.path.join(args.run_dir, f"{args.model_id}.pt")
    else:
        checkpoint_path = f"wandb/latest-run/{args.model_id}.pt"
    evaluate_model(
        checkpoint_path=checkpoint_path,
        model_id=args.model_id,
        num_episodes=args.num_episodes,
        record_video=not args.no_video,
        run_dir=args.run_dir,
        project_name=args.project_name
    )

    # Example usage:
    # python agilerl_maddpg_eval.py _0_10000 --num_episodes 5 --no_video