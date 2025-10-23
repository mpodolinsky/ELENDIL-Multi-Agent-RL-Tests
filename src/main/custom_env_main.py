# src/main/custom_env_main.py
import yaml
import time
from gymnasium_env.envs.grid_world_multi_agent import GridWorldEnvMultiAgent

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
import wandb
from wandb.integration.sb3 import WandbCallback

# Import the observation wrapper
from observation_wrapper import FlattenMultiAgentWrapper

# Config file paths

agent_config_path = "configs/agent_config.yaml"
target_config_path = "configs/target_config.yaml"
env_config_path = "configs/env_config.yaml"

agents: list[dict] = []
target_config: dict = {}
env_config: dict = {}

# Load configurations from YAML files

with open(agent_config_path, "r") as f:
    agent_config = yaml.safe_load(f)
    agents.append({**agent_config})

with open(target_config_path, "r") as f:
    target_config = yaml.safe_load(f)
    target_config = {**target_config}

with open(env_config_path, "r") as f:
    env_config = yaml.safe_load(f)
    env_config = {**env_config}

# WandB configuration

config = {
    "env_name":             "GridWorldEnvMultiAgent",
    "total_timesteps":      250000,
    "num_envs":             8,
    "env_config":           {**env_config},
    "agent_config":         {**agent_config},
    "target_config":        {**target_config},
    "algorithm":            "PPO",
    "type":                 "default-sb3",
    "policy_type":          "MlpPolicy",
}

# WandB initialization

run = wandb.init(
    project=            "SingleAgentStealth-v0.1",
    name=               f"simple_test_1a1t_l0.5_obstacles_3_hidden_target_size_7{time.strftime('%Y%m%d-%H%M%S')}",
    config=             config,
    sync_tensorboard=   True,  # auto-upload sb3's tensorboard metrics
    monitor_gym=        True,  # auto-upload the videos of agents playing the game
    save_code=          True,  # optional
)

def make_env():
    env = GridWorldEnvMultiAgent(render_mode=           wandb.config.env_config["render_mode"],
                                size=                   wandb.config.env_config["size"],
                                agents=                 [wandb.config.agent_config],  # Wrap in list
                                no_target=              wandb.config.env_config["no_target"],
                                enable_obstacles=       wandb.config.env_config["enable_obstacles"],
                                num_obstacles=          wandb.config.env_config["num_obstacles"],
                                show_fov_display=       wandb.config.env_config["show_fov_display"],
                                target_config=          wandb.config.target_config,
                                lambda_fov=             wandb.config.env_config["lambda_fov"],
                                max_steps=              wandb.config.env_config["max_steps"],)
    
    # Wrap with multi-agent flattener to handle nested observation and action spaces
    env = FlattenMultiAgentWrapper(env)
    env = Monitor(env)  # record stats such as returns
    return env

# Create vectorized environment and wrap it for video recording
env = DummyVecEnv([make_env])
env = VecVideoRecorder(
    env,
    f"wandb/latest-run/videos/{run.id}",
    record_video_trigger=lambda x: x % 20000 == 0,
    video_length=200,
)

# Initialize Model
model = PPO(config["policy_type"], env, verbose=1, tensorboard_log=f"runs/{run.id}")

# Train the model
model.learn(
    total_timesteps=config["total_timesteps"],
    callback=WandbCallback(
        gradient_save_freq=100,
        model_save_path=f"models/{run.id}",
        verbose=2,
    ),
)
run.finish()

