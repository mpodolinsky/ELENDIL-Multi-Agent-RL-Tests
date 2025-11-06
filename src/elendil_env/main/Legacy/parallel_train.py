"""
Training script using PettingZoo Parallel environment with SuperSuit for SB3 compatibility.
This approach uses native PettingZoo parallel execution instead of custom wrappers.
"""
import yaml
import time
import os
import glob
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CallbackList
import wandb
from wandb.integration.sb3 import WandbCallback

# Import the parallel environment
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from envs.parallel_gridworld import parallel_env

###################################################################
# Before launching the training, make sure to change if necessary:

# 1. YAML files
# 2. Run tags
# 3. Run name
# 4. Run notes
# 5. RL Algorithm

###################################################################

# Configuration file paths
ground_agent_config_path = "configs/agent_configs/ground_agent.yaml"
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

# Create list of agent configurations
agent_configs = [ground_agent_config]

# WandB configuration
config = {
    "env_name":             "GridWorldEnvMultiAgent_Parallel",
    "total_timesteps":      250000,
    "num_envs":             8,  # Number of parallel environments
    "env_config":           {**env_config},
    "agent_config":         {**ground_agent_config},
    "target_config":        {**target_config},
    "algorithm":            "PPO",
    "type":                 "supersuit-parallel",
    "policy_type":          "MlpPolicy",
}

if __name__ == '__main__':
    # WandB initialization
    run = wandb.init(
        project="SingleAgentStealth-v0.1",
        name=f"HA_SPO2V_parallel_medium_env_1g1t_ppo_{config['num_envs']}_envs_{time.strftime('%Y%m%d-%H%M%S')}",
        tags=["medium_env_obstacles", "parallel", "1g1t", "ppo", f"{config['num_envs']} envs"],
        notes="Parallel environment using PettingZoo + SuperSuit, 1 ground agent, 1 target, medium env.",
        config=config,
        sync_tensorboard=True,
        monitor_gym=True,
        save_code=True,
    )
    
    print("Creating parallel environment...")
    
    # Create the base parallel environment
    def make_env():
        return parallel_env(
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
    
    # Create a single environment and convert to SB3 format
    base_env = make_env()
    
    # Convert PettingZoo parallel env to vectorized env for SB3
    vec_env = pettingzoo_env_to_vec_env_v1(base_env)
    
    # Concatenate multiple environments for parallel training
    num_envs = config["num_envs"]
    if num_envs > 1:
        vec_env = concat_vec_envs_v1(
            vec_env, 
            num_vec_envs=num_envs, 
            num_cpus=num_envs,
            base_class='subproc'  # Use multiprocessing for true parallelism
        )
    
    print(f"Environment created with {num_envs} parallel instances")
    print(f"Observation space: {vec_env.observation_space}")
    print(f"Action space: {vec_env.action_space}")
    
    # Initialize PPO model
    model = PPO(
        config["policy_type"], 
        vec_env, 
        verbose=1, 
        tensorboard_log=f"runs/{run.id}"
    )
    
    print("Starting training...")
    
    # Train the model
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"wandb/latest-run/models/{run.id}",
            verbose=2,
        ),
    )
    
    # Save final model
    final_model_path = f"wandb/latest-run/models/{run.id}/final_model.zip"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save(final_model_path)
    print(f"Final model saved to {final_model_path}")
    
    # Upload final model to WandB
    if os.path.exists(final_model_path):
        wandb.save(final_model_path, base_path=os.path.dirname(final_model_path))
        
    # Upload all model checkpoints to WandB
    model_dir = f"wandb/latest-run/models/{run.id}"
    if os.path.exists(model_dir):
        checkpoint_files = glob.glob(os.path.join(model_dir, "*.zip"))
        for checkpoint_path in checkpoint_files:
            wandb.save(checkpoint_path, base_path=model_dir)
    
    run.finish()
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("="*60)

