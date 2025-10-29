"""This tutorial shows how to train an MADDPG agent on the Elendil GridWorld environment.

Based on the AgileRL MADDPG tutorial for PettingZoo.
"""

import os
from copy import deepcopy

import numpy as np
import torch
import yaml
import wandb
import time
from agilerl.algorithms.core.registry import HyperparameterConfig, RLParameter
from agilerl.components.multi_agent_replay_buffer import MultiAgentReplayBuffer
from agilerl.hpo.mutation import Mutations
from agilerl.hpo.tournament import TournamentSelection
from agilerl.training.train_multi_agent_off_policy import train_multi_agent_off_policy
from agilerl.utils.utils import create_population
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv
from tqdm import trange

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

if __name__ == "__main__":
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

    # WandB configuration
    config = {
        "env_name": "GridWorldEnvParallel",
        "total_timesteps": 1_000_000,
        "num_envs": 4,
        "env_config": {**env_config},
        "agent_config": agent_configs[0],
        "target_config": {**target_config},
        "algorithm": "MADDPG",
        "type": "elendil-agilerl-maddpg-train-func",
    }
    
    # TODO write a function that checks for a predefined list of tags that are allowed

    # tags = [....]
    # if check_tags(tags):
    #     print("Tags are allowed")
    # else:
    #     print("Tags are not allowed")
    #     exit()
    
    # WandB initialization
    run = wandb.init(
        project="ELENDIL",
        name=f"ELENDIL_agilerl_maddpg_train_func_{time.strftime('%Y%m%d-%H%M%S')}",
        tags=["medium_env_obstacles", "elendil", "parallel", "1g1a1t", "maddpg", "agilerl", "4_envs", "default_hp_agilerl", "seed=1"],
        notes="ELENDIL environment using AgileRL's MADDPG with train_multi_agent_off_policy function, observation flattening wrapper, 1 ground agent, 1 air observer agent, 1 target, medium env.",
        config=config,
        save_code=True,
    )

    # Defaults from https://docs.agilerl.com/en/latest/tutorials/pettingzoo/maddpg.html
    # Define the initial hyperparameters
    INIT_HP = {
        "POPULATION_SIZE": 1,   # Population size
        "ALGO": "MADDPG",       # Algorithm
        "CHANNELS_LAST": False, # Not using image observations
        "BATCH_SIZE": 128,      # Batch size
        "O_U_NOISE": True,      # Ornstein Uhlenbeck action noise
        "EXPL_NOISE": 0.1,      # Action noise scale
        "MEAN_NOISE": 0.0,      # Mean action noise
        "THETA": 0.15,          # Rate of mean reversion in OU noise
        "DT": 0.01,             # Timestep for OU noise
        "LR_ACTOR": 0.001,      # Actor learning rate                  # CUSTOM, Default 0.0001
        "LR_CRITIC": 0.001,     # Critic learning rate
        "GAMMA": 0.95,          # Discount factor
        "MEMORY_SIZE": 100_000, # Max memory buffer size
        "LEARN_STEP": 1,        # Learning frequency                   # CUSTOM, Default 50
        "TAU": 0.01,            # For soft update of target parameters
    }

    num_envs = wandb.config["num_envs"]
    print(f"Creating {num_envs} environments...")

    # Define environment factory function
    def make_env(render_mode=None):
        """Factory function to create a new environment instance."""
        return GridWorldEnvParallel(
            size=env_config["size"],
            max_steps=env_config["max_steps"],
            render_mode=render_mode,
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

    # Create vectorized environment wrapped with observation flatten wrapper
    # Wrap each environment in the AsyncVecEnv
    def _make_wrapped_env():
        base = make_env()
        return ObservationFlattenWrapper(base)
    
    env = AsyncPettingZooVecEnv([_make_wrapped_env for _ in range(num_envs)])
    
    # Note: AgileRL's train_multi_agent_off_policy doesn't support video recording
    # For visualization, you'll need to use the eval script: src/elendil/test/agilerl_maddpg_eval.py
    print("Note: Use agilerl_maddpg_eval.py to generate videos after training")
    
    env.reset()

    # Configure the multi-agent algo input arguments
    observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
    action_spaces = [env.single_action_space(agent) for agent in env.agents]

    print(f"Observation spaces: {observation_spaces}")
    print(f"Action spaces: {action_spaces}")

    # Append number of agents and agent IDs to the initial hyperparameter dictionary
    INIT_HP["AGENT_IDS"] = env.agents
    
    # Network configuration for Box observation spaces
    # When using Box spaces (after flattening), AgileRL expects encoder_config
    NET_CONFIG = None  # Let AgileRL auto-configure for Box spaces

    # Default from https://docs.agilerl.com/en/latest/tutorials/pettingzoo/matd3.html
    # Mutation config for RL hyperparameters
    hp_config = HyperparameterConfig(
        lr_actor=RLParameter(min=1e-4, max=1e-2),
        lr_critic=RLParameter(min=1e-4, max=1e-2),
        batch_size=RLParameter(min=8, max=512, dtype=int),
        # learn_step=RLParameter(min=20, max=200, dtype=int),
        gamma=RLParameter(min=0.90, max=0.99),
    )

    # Defaults from https://docs.agilerl.com/en/latest/tutorials/pettingzoo/matd3.html
    # Create mutations and tournament selection
    # Note: For population_size=1, mutations may not be needed
    mutations = Mutations(
        no_mutation=0.2,
        architecture=0.2,
        new_layer_prob=0.2,
        parameters=0.2,
        activation=0,
        rl_hp=0.2,
        mutation_sd=0.1,
        mutate_elite=True,
        device=device,
        rand_seed=1 # For reproducibility during testing
    )
    
    tournament = TournamentSelection(
        population_size=INIT_HP["POPULATION_SIZE"],
        tournament_size=2,
        elitism=True,
        eval_loop=1,
    )

    # Create population of agents
    print(f"Creating MADDPG population...")
    pop = create_population(
        algo=INIT_HP["ALGO"],
        observation_space=observation_spaces,
        action_space=action_spaces,
        net_config=NET_CONFIG,
        INIT_HP=INIT_HP,
        population_size=INIT_HP["POPULATION_SIZE"],
        num_envs=num_envs,
        device=device,
    )

    agent = pop[0]

    # Create experience replay buffer
    memory = MultiAgentReplayBuffer(
        memory_size=INIT_HP["MEMORY_SIZE"],
        field_names=["state", "action", "reward", "next_state", "done"],
        agent_ids=env.agents,
        device=device,
    )

    checkpoint_path_1 = "wandb/latest-run/"
    os.makedirs(checkpoint_path_1, exist_ok=True)
    print(f"Checkpoint directory created: {checkpoint_path_1}")

    # Defaults from https://docs.agilerl.com/en/latest/tutorials/pettingzoo/matd3.html
    # Use the provided training function
    # Note: Checkpointing in train_multi_agent_off_policy is currently buggy, so we disable it
    # and save manually after training
    print(f"Starting training...")
    try:
        trained_pop, pop_fitnesses = train_multi_agent_off_policy(
            env=env,
            env_name="elendil_gridworld",
            algo="MADDPG",
            pop=pop,
            memory=memory,
            INIT_HP=INIT_HP,
            max_steps=wandb.config["total_timesteps"],  # Start with fewer steps for testing
            evo_steps=10_000,   # Evolution frequency
            eval_steps=None,  # Let it use default evaluation, go until done
            eval_loop=10,  # Number of evaluation episodes CUSTOM for stability, Default 1
            learning_delay=500,  # Steps before starting learning
            target=None,  # No early stopping
            tournament=tournament,
            mutation=mutations,
            wb=True,  # Enable WandB logging
            # Disable checkpointing in train function - save manually instead
            checkpoint=int(wandb.config["total_timesteps"]/10),
            checkpoint_path=checkpoint_path_1,
        )
        
        # Save final model
        save_path = f"wandb/latest-run/models/{run.id}/"
        os.makedirs(save_path, exist_ok=True)
        final_path = os.path.join(save_path, "final_model.pt")
        trained_pop[0].save_checkpoint(final_path)
        print(f"\n Training complete! Saved final model to {final_path}")
        
    except KeyboardInterrupt:
        print("Training interrupted by user")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Finish WandB run
        wandb.finish()
        print("Training completed and logged to WandB")

    env.close()
