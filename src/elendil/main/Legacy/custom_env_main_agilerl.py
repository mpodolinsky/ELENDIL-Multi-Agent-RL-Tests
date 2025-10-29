# src/elendil/main/custom_env_main_agilerl.py
import yaml
import time
import os
import glob
import sys
import numpy as np

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
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.utils.utils import create_population
import torch

import wandb

import gymnasium as gym

###################################################################
# Before launching the training, make sure to change if necessary:
#
# 1. YAML files
# 2. Run tags
# 3. Run name
# 4. Run notes
# 5. RL Algorithm
###################################################################
 
# Configuration file paths
ground_agent_config_path = "configs/agent_configs/ground_agent.yaml"
air_observer_config_path = "configs/agent_configs/air_observer_agent.yaml"
target_config_path = "configs/target_configs/target_config.yaml"
env_config_path = "configs/env_configs/medium_env_obstacles.yaml"

with open(env_config_path, "r") as f:
    env_config = yaml.safe_load(f)

# Load target configuration
with open(target_config_path, "r") as f:
    target_config = yaml.safe_load(f)

# Load agent configurations from YAML files
with open(ground_agent_config_path, "r") as f:
    ground_agent_config = yaml.safe_load(f)

with open(air_observer_config_path, "r") as f:
    air_observer_config = yaml.safe_load(f)

# Create list of agent configurations
# The environment will automatically instantiate agents from these configs!
agent_configs = [
    ground_agent_config,
    air_observer_config,
]

# WandB configuration
config = {
    "env_name":             "GridWorldEnvParallel",
    "total_timesteps":      1_000_000,
    "num_envs":             4,
    "env_config":           {**env_config},
    "agent_config":         {**ground_agent_config},
    "target_config":        {**target_config},
    "algorithm":            "MADDPG",
    "type":                 "elendil-agilerl-maddpg",
}

# TAGS
# - "small_env"
# - "medium_env"
# - "large_env"
# - "small_env_obstacles"
# - "medium_env_obstacles"
# - "large_env_obstacles"
# - "baseline"
# - "mappo"
# - "agilerl"

# ========================================
# AgileRL MAPPO Setup
# ========================================
if __name__ == '__main__':
    # WandB initialization
    run = wandb.init(
        project=            "ELENDIL",
        name=               f"ELENDIL_agilerl_maddpg_4_envs_medium_obstacles_1a1t_{time.strftime('%Y%m%d-%H%M%S')}",
        tags=               ["medium_env_obstacles", "elendil", "parallel", "1g1a1t", "maddpg", "agilerl", "4_envs"],
        notes=              "ELENDIL environment using AgileRL's MADDPG implementation with parallel PettingZoo environment, 1 ground agent, 1 air observer agent, 1 target, medium env.",
        config=             config,
        save_code=          True,
    )
    
    # Create the parallel environment
    print("🏗️  Creating ELENDIL parallel environment...")
    env = GridWorldEnvParallel(
        # Size and steps
        size=env_config["size"],
        max_steps=env_config["max_steps"],
        
        # Rendering
        render_mode=env_config["render_mode"],
        show_fov_display=env_config["show_fov_display"],
        
        # Rewards
        intrinsic=env_config.get("intrinsic", False),
        lambda_fov=env_config["lambda_fov"],
        show_target_coords=env_config.get("show_target_coords", False),
        
        # Targets and agents
        no_target=env_config["no_target"],
        agents=agent_configs,
        target_config=target_config,
        
        # Obstacles
        enable_obstacles=env_config["enable_obstacles"],
        num_obstacles=env_config["num_obstacles"],
        num_visual_obstacles=env_config["num_visual_obstacles"],
    )
    
    print(f"✅ Environment created")
    print(f"✅ Agents: {env.agents}")
    print(f"✅ Observation spaces: {env.observation_space(env.agents[0])}")
    print(f"✅ Action space: {env.action_space(env.agents[0])}")
    
    # Create multiple parallel environments for vectorization
    num_envs = config["num_envs"]
    
    def make_env():
        """Factory function to create a new environment instance."""
        return GridWorldEnvParallel(
            size=env_config["size"],
            max_steps=env_config["max_steps"],
            render_mode=env_config["render_mode"],
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
    
    # Create list of environment factory functions wrapped with ObservationFlattenWrapper
    def _make_wrapped_env():
        base = make_env()
        return ObservationFlattenWrapper(base)
    
    env = AsyncPettingZooVecEnv([_make_wrapped_env for _ in range(num_envs)])
    
    print(f"✅ Wrapped {num_envs} environments for AgileRL with observation flattening")
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"📱 Using device: {device}")
    
    print(f"📊 Observation space: {env.observation_space}")
    print(f"📊 Action space: {env.action_space}")
    print(f"📊 Possible agents: {env.possible_agents}")
    
    # Store agent IDs in a variable first
    agent_ids = env.possible_agents
    
    # Define network configuration for Box observations (after flattening)
    # Since we're using ObservationFlattenWrapper, we now have Box spaces
    NET_CONFIG = None  # Let AgileRL auto-configure for Box spaces
    
    # Initialize hyperparameters (standard AgileRL MADDPG defaults)
    INIT_HP = {
        "ALGO": "MADDPG",
        "BATCH_SIZE": 128,         # Standard AgileRL batch size
        "LR_ACTOR": 0.001,        # Standard learning rates
        "LR_CRITIC": 0.001,
        "GAMMA": 0.99,            # Standard discount factor
        "MEMORY_SIZE": 100000,    # Standard replay buffer size
        "LEARN_STEP": 128,        # Standard learn frequency
        "TAU": 0.01,              # Standard soft update parameter
        "EXPL_NOISE": 0.1,       # Standard exploration noise
        "O_U_NOISE": True,
        "MEAN_NOISE": 0.0,
        "THETA": 0.15,
        "DT": 0.01,
        "POP_SIZE": 4,           # Standard population size
        "AGENT_IDS": agent_ids,  # Required for MADDPG multi-agent training
    }
    
    # Create population of agents
    print("🏗️  Creating MADDPG agents...")
    
    # Get observation and action spaces for all agents (as lists, as per AgileRL docs)
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]
    
    print(f"📊 Observation spaces: {observation_spaces}")
    print(f"📊 Action spaces: {action_spaces}")
    
    # Create single MADDPG agent (not using create_population)
    # Initialize MADDPG directly as per tutorial
    agent = MADDPG(
        observation_spaces=observation_spaces,
        action_spaces=action_spaces,
        agent_ids=agent_ids,
        net_config=None,
        batch_size=INIT_HP["BATCH_SIZE"],
        lr_actor=INIT_HP["LR_ACTOR"],
        lr_critic=INIT_HP["LR_CRITIC"],
        gamma=INIT_HP["GAMMA"],
        tau=INIT_HP["TAU"],
        O_U_noise=INIT_HP["O_U_NOISE"],
        expl_noise=INIT_HP["EXPL_NOISE"],
        mean_noise=INIT_HP["MEAN_NOISE"],
        theta=INIT_HP["THETA"],
        dt=INIT_HP["DT"],
        learn_step=INIT_HP["LEARN_STEP"],
        normalize_images=False,  # Disable image normalization for Dict spaces
        device=device,
    )
    agent.set_training_mode(True)
    
    agents = [agent]  # Wrap in list for consistency
    
    print(f"✅ Created {len(agents)} agents")
    
    # Create replay buffer
    replay_buffer = MultiAgentReplayBuffer(
        memory_size=INIT_HP["MEMORY_SIZE"],
        field_names=["state", "action", "reward", "next_state", "done"],
        agent_ids=agent_ids,
        device=device,
    )
    
    # Set up evolutionary components (for multi-agent training)
    mutations = Mutations(
        no_mutation=0.4,
        architecture=0.2,
        new_layer_prob=0.2,
        parameters=0.2,
        activation=0.2,
        rl_hp=0.1,
        mutation_sd=0.1,
    )
    selection = TournamentSelection(
        tournament_size=2, 
        elitism=True,
        population_size=INIT_HP["POP_SIZE"],
        eval_loop=1
    )
    
    print(f"📋 Population size: {INIT_HP['POP_SIZE']}")
    print(f"📋 Batch size: {INIT_HP['BATCH_SIZE']}")
    print(f"📋 Learn step: {INIT_HP['LEARN_STEP']}")
    
    # Training configuration
    max_steps = config["total_timesteps"]
    total_episodes = max_steps // env_config["max_steps"]
    save_frequency = 1000  # Save model every N episodes
    
    print(f"🎯 Starting training for {total_episodes} episodes ({max_steps} steps)")
    
    # Training loop - track steps and resets properly
    episode_count = 0
    step_count = 0
    
    # Track scores for averaging (required by AgileRL)
    if not hasattr(agent, 'scores'):
        agent.scores = []
    
    try:
        # Get initial observations from vectorized environment
        observations, info = env.reset()
        
        episode_rewards = {agent_id: [0] * num_envs for agent_id in env.agents}
        completed_episodes = {i: False for i in range(num_envs)}
        
        while step_count < max_steps:
            # Get actions from each agent
            # Note: `agents` is a single agent object, not a list
            # Use `agents[0]` to get the first agent in the population
            agent_obj = agents[0] if isinstance(agents, list) else agents
            
            actions, raw_actions = agent_obj.get_action(obs=observations, infos=info)
            
            # Step environment
            next_observations, rewards, terminations, truncations, infos = env.step(actions)
            
            # Store transitions in replay buffer
            # Note: observations, actions, rewards are dictionaries keyed by agent
            replay_buffer.save_to_memory(
                observations,
                raw_actions,  # Use raw actions from get_action
                rewards,
                next_observations,
                terminations,
                is_vectorised=True,
            )
            
            # Sum rewards for logging
            for agent_id in env.possible_agents:
                for env_idx in range(num_envs):
                    episode_rewards[agent_id][env_idx] += rewards[agent_id][env_idx]
            
            observations = next_observations
            step_count += num_envs
            
            # Check which environments have completed episodes
            for env_idx in range(num_envs):
                if not completed_episodes[env_idx]:
                    # Check if ANY agent in this env is done
                    for agent_id in env.possible_agents:
                        term_vals = terminations.get(agent_id, np.array([False] * num_envs))
                        trunc_vals = truncations.get(agent_id, np.array([False] * num_envs))
                        
                        # Handle both dict and array returns
                        if isinstance(term_vals, np.ndarray):
                            term_val = term_vals[env_idx] if env_idx < len(term_vals) else False
                        elif isinstance(term_vals, dict):
                            term_val = term_vals.get(env_idx, False)
                        else:
                            term_val = False
                        
                        if isinstance(trunc_vals, np.ndarray):
                            trunc_val = trunc_vals[env_idx] if env_idx < len(trunc_vals) else False
                        elif isinstance(trunc_vals, dict):
                            trunc_val = trunc_vals.get(env_idx, False)
                        else:
                            trunc_val = False
                        
                        if term_val or trunc_val:
                            # Mark episode as completed
                            completed_episodes[env_idx] = True
                            episode_count += 1
                            
                            # Log completed episode
                            avg_rewards = {}
                            for agent_id in env.possible_agents:
                                avg_rewards[agent_id] = episode_rewards[agent_id][env_idx]
                            
                            # Calculate total score (sum of all agent rewards)
                            total_score = sum(avg_rewards.values())
                            
                            # Track score for averaging (used by AgileRL evaluation metrics)
                            agent.scores.append(total_score)
                            
                            print(f"✅ Episode {episode_count} completed in env {env_idx}: {avg_rewards} (total score: {total_score})")
                            
                            # Log to WandB - include both individual rewards and total score
                            wandb.log({
                                "steps": step_count,
                                "episode": episode_count,
                                "score": total_score,  # Total score (sum of all agent rewards)
                                **{f"reward_{agent}": avg_rewards[agent] for agent in env.possible_agents}
                            })
                            
                            # Save model periodically
                            if episode_count % save_frequency == 0:
                                save_path = f"wandb/latest-run/models/{run.id}/maddpg_episode_{episode_count}.pt"
                                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                                agent.save_checkpoint(save_path)
                                print(f"✅ Saved model at episode {episode_count}")
                            
                            # Reset rewards for this env
                            for agent_id in env.possible_agents:
                                episode_rewards[agent_id][env_idx] = 0
                            
                            break
            
            # Reset environments that completed episodes
            if any(completed_episodes.values()):
                # Reset completed environments
                observations, info = env.reset()
                completed_episodes = {i: False for i in range(num_envs)}
                # Reset episode rewards
                episode_rewards = {agent_id: [0] * num_envs for agent_id in env.agents}
            
            # Learn from experience (standard AgileRL learn frequency)
            if len(replay_buffer) > INIT_HP["BATCH_SIZE"] and step_count % INIT_HP["LEARN_STEP"] == 0:
                # Sample from replay buffer
                experiences = replay_buffer.sample(INIT_HP["BATCH_SIZE"])
                # Learn from experiences
                agent.learn(experiences)
                
    except KeyboardInterrupt:
        print("⚠️  Training interrupted by user")
    except Exception as e:
        print(f"❌ Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Save final model
        final_model_path = f"wandb/latest-run/models/{run.id}/maddpg_final.pt"
        os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
        agent.save_checkpoint(final_model_path)
        print(f"✅ Saved final model")
        
        wandb.finish()
        print("✅ Training completed and logged to WandB")

