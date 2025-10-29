# src/main/custom_env_main.py
import yaml
import time
import os
import glob
import sys

# Add ELENDIL package to Python path
elendil_path = "/mnt/data/Documents/Project_M/ELENDIL"
if elendil_path not in sys.path:
    sys.path.insert(0, elendil_path)

from elendil.envs.grid_world_multi_agent import GridWorldEnvParallel

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecVideoRecorder, SubprocVecEnv, DummyVecEnv
import wandb
from wandb.integration.sb3 import WandbCallback

import gymnasium as gym
import numpy as np
from gymnasium import spaces

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
    # ground_agent_config
    air_observer_config
]

# WandB configuration

config = {
    "env_name":             "GridWorldEnvParallel",
    "total_timesteps":      1_000_000,  # Reduced for stability testing
    "num_envs":             4,                           # Reduced to 2 for memory optimization
    "env_config":           {**env_config},
    "agent_config":         {**ground_agent_config},
    "target_config":        {**target_config},
    "algorithm":            "PPO",
    "type":                 "elendil-parallel-wrapper",
    "policy_type":          "MultiInputPolicy",          # Required for dict observation spaces
}

# TAGS

# - "small_env"
# - "medium_env"
# - "large_env"
# - "small_env_obstacles"
# - "medium_env_obstacles"
# - "large_env_obstacles"

# - "baseline"
# - "ippo"
# - "ppo"
# - "a2c"
# - "dqn"
# - "ddpg"
# - "td3"
# - "sac"
# - "trpo"

# - "1g1t"
# - "1a2t"
# - "1a3t"
# - "1a1t"
# - "1a2t"
# - "1a3t"

# - "8 envs"

# ========================================
# PettingZoo Parallel to Gymnasium Wrapper
# ========================================

class PettingZooParallelToGymnasium(gym.Env):
    """
    Wrapper to convert PettingZoo ParallelEnv to Gymnasium Env for Stable-Baselines3 compatibility.
    """
    
    def __init__(self, parallel_env):
        super().__init__()
        self.parallel_env = parallel_env
        
        # Convert observation and action spaces to Gymnasium format
        # For single agent, we'll use the first agent's spaces
        if hasattr(parallel_env, 'observation_spaces') and parallel_env.observation_spaces:
            first_agent = list(parallel_env.observation_spaces.keys())[0]
            self.observation_space = parallel_env.observation_spaces[first_agent]
        else:
            # Fallback to a default space
            self.observation_space = spaces.Box(low=0, high=1, shape=(10,), dtype=np.float32)
        
        if hasattr(parallel_env, 'action_spaces') and parallel_env.action_spaces:
            first_agent = list(parallel_env.action_spaces.keys())[0]
            self.action_space = parallel_env.action_spaces[first_agent]
        else:
            # Fallback to a default space
            self.action_space = spaces.Discrete(5)
        
        # Store the first agent ID for action/observation mapping
        self.first_agent = list(parallel_env.agents)[0] if parallel_env.agents else None
        
        # Preserve render_mode
        if hasattr(parallel_env, 'render_mode'):
            self.render_mode = parallel_env.render_mode
        else:
            self.render_mode = None
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            # PettingZoo parallel envs might not support seed in reset
            try:
                observations, infos = self.parallel_env.reset(seed=seed)
            except TypeError:
                observations, infos = self.parallel_env.reset()
        else:
            observations, infos = self.parallel_env.reset()
        
        # Return observation for the first agent
        if self.first_agent and self.first_agent in observations:
            obs = observations[self.first_agent]
        else:
            # Fallback if agent not found
            obs = list(observations.values())[0] if observations else np.zeros(10)
        
        return obs, {}
    
    def step(self, action):
        """Step the environment."""
        # Convert single action to multi-agent action dict
        if self.first_agent:
            actions = {self.first_agent: action}
        else:
            # Fallback: use first available agent
            first_agent = list(self.parallel_env.agents)[0]
            actions = {first_agent: action}
        
        # Step the parallel environment
        observations, rewards, terminations, truncations, infos = self.parallel_env.step(actions)
        
        # Extract values for the first agent
        if self.first_agent and self.first_agent in observations:
            obs = observations[self.first_agent]
            reward = rewards[self.first_agent]
            terminated = terminations[self.first_agent]
            truncated = truncations[self.first_agent]
            info = infos[self.first_agent] if self.first_agent in infos else {}
        else:
            # Fallback
            first_agent = list(observations.keys())[0]
            obs = observations[first_agent]
            reward = rewards[first_agent]
            terminated = terminations[first_agent]
            truncated = truncations[first_agent]
            info = infos[first_agent] if first_agent in infos else {}
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if hasattr(self.parallel_env, 'render'):
            return self.parallel_env.render()
    
    def close(self):
        """Close the environment."""
        if hasattr(self.parallel_env, 'close'):
            self.parallel_env.close()

# ========================================
# Function Definitions (importable by child processes)
# ========================================

def make_single_env():
    """
    Factory function to create a single parallel environment instance.
    This function doesn't depend on wandb.config, so it works in child processes.
    """
    def _init():
        # Create the parallel environment directly
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
        
        # Convert PettingZoo parallel environment to Gymnasium environment
        env = PettingZooParallelToGymnasium(env)
        
        return env
    return _init


# ========================================
# Main Execution (protected from child processes)
# ========================================
if __name__ == '__main__':
    # This block only runs in the main process, not in child processes spawned by SubprocVecEnv
    
    # WandB initialization
    run = wandb.init(
        project=            "ELENDIL",
        name=               f"ELENDIL_parallel_4_envs_medium_obstacles_1a1t_ppo_{time.strftime('%Y%m%d-%H%M%S')}",
        tags=               ["medium_env_obstacles", "elendil", "parallel", "1a1t", "ppo", "4_envs", "supersuit"],
        notes=              "ELENDIL parallel environment with SuperSuit vectorization (single env), 1 air observer agent, 1 target, medium env, PPO, using the standard 1-cell detection method.",
        config=             config,
        sync_tensorboard=   True,  # auto-upload sb3's tensorboard metrics
        monitor_gym=        True,  # auto-upload the videos of agents playing the game
        save_code=          True,  # optional
    )
    
    # ========================================
    # ELENDIL Parallel Environment Setup
    # ========================================
    # This approach uses:
    # - ELENDIL's GridWorldEnvParallel (PettingZoo parallel environment)
    # - Custom PettingZooParallelToGymnasium wrapper (converts to Gymnasium)
    # - Stable-Baselines3's make_vec_env (handles multiprocessing)
    # 
    # Benefits:
    # - Native PettingZoo parallel environment support
    # - Automatic multiprocessing with SubprocVecEnv
    # - Clean integration with Stable-Baselines3
    # - Proper Monitor wrapping for training stats
    # ========================================
    
    num_envs = config["num_envs"]
    
    # Create vectorized environment using Stable-Baselines3's make_vec_env
    # This handles multiprocessing automatically and works with our Gymnasium wrapper
    env = make_vec_env(
        make_single_env(),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv if num_envs > 1 else DummyVecEnv,
        monitor_dir=f"wandb/latest-run/monitors/{run.id}",  # Save monitoring data
    )
    
    print(f"✅ Created vectorized environment with {num_envs} environments")
    print(f"✅ Observation space: {env.observation_space}")
    print(f"✅ Action space: {env.action_space}")
    
    # Wrap for video recording
    # Calculate video recording frequency (every 1/20th of total episodes for more frequent recording)
    episode_steps = env_config["max_steps"]
    total_timesteps = config["total_timesteps"]
    num_episodes = total_timesteps // (episode_steps * num_envs)
    record_interval = max(1, num_episodes // 20) * episode_steps * num_envs  # More frequent recording
    
    # Memory optimization: Reduce video recording frequency to save memory
    record_interval = max(record_interval, 20000)  # Minimum 20k steps between videos
    
    env = VecVideoRecorder(
        env,
        f"wandb/latest-run/videos/{run.id}",
        record_video_trigger=lambda x: x % record_interval == 0,
        video_length=episode_steps,
        name_prefix=f"training-video"
    )
    
    print(f"✅ Video recording enabled (every {record_interval} steps)")
    print(f"📊 Expected videos by end of training: {total_timesteps // record_interval}")
    
    # Initialize Model
    model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")
    
    # Add memory monitoring
    import psutil
    import gc
    
    def log_memory_usage():
        memory = psutil.virtual_memory()
        print(f"🔍 Memory usage: {memory.percent:.1f}% ({memory.used/1024**3:.1f}GB/{memory.total/1024**3:.1f}GB)")
        if memory.percent > 85:
            print("⚠️  WARNING: High memory usage detected!")
            gc.collect()  # Force garbage collection
            print("🧹 Garbage collection performed")
    
    log_memory_usage()
    
    # Create custom callback for memory monitoring
    from stable_baselines3.common.callbacks import BaseCallback
    
    class MemoryCallback(BaseCallback):
        def __init__(self, check_freq: int, verbose=1):
            super(MemoryCallback, self).__init__(verbose)
            self.check_freq = check_freq
            self.memory_threshold = 90  # Kill if memory > 90%
            
        def _on_step(self) -> bool:
            if self.n_calls % self.check_freq == 0:
                memory = psutil.virtual_memory()
                if self.verbose > 0:
                    print(f"🔍 Step {self.n_calls}: Memory {memory.percent:.1f}%")
                
                if memory.percent > self.memory_threshold:
                    print(f"🚨 CRITICAL: Memory usage {memory.percent:.1f}% exceeds threshold!")
                    print("🛑 Stopping training to prevent system crash...")
                    return False  # Stop training
                
                # Force garbage collection every 1000 steps
                if self.n_calls % 1000 == 0:
                    gc.collect()
                    if self.verbose > 0:
                        print("🧹 Garbage collection performed")
            
            return True
    
    # Train the model with memory monitoring
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=[
            WandbCallback(
                gradient_save_freq=100,  # Very frequent checkpoints
                model_save_path=f"wandb/latest-run/models/{run.id}",
                verbose=2,
            ),
            MemoryCallback(check_freq=100, verbose=1)  # Check memory every 100 steps
        ],
    )
    
    # Save final model
    final_model_path = f"wandb/latest-run/models/{run.id}/final_model.zip"
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save(final_model_path)
    
    # Upload videos to WandB as artifacts
    video_dir = f"wandb/latest-run/videos/{run.id}"
    if os.path.exists(video_dir):
        video_files = glob.glob(os.path.join(video_dir, "*.mp4"))
        if video_files:
            # Create a WandB artifact for training videos
            artifact = wandb.Artifact(
                name=f"training_videos_{run.id}",
                type="video",
                description=f"Training videos from ELENDIL environment run {run.id}"
            )
            
            # Add all video files to the artifact
            for video_path in video_files:
                video_name = os.path.basename(video_path)
                artifact.add_file(video_path, name=video_name)
                print(f"✅ Added video to artifact: {video_name}")
            
            # Log the artifact
            run.log_artifact(artifact)
            print(f"✅ Uploaded {len(video_files)} videos as WandB artifact")
            
            # Also log individual videos for immediate viewing
            for video_path in video_files:
                video_name = os.path.basename(video_path)
                wandb.log({f"videos/{video_name}": wandb.Video(video_path, fps=4, format="mp4")})
    
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

