"""This tutorial shows how to train an MADDPG agent on the Elendil GridWorld environment.

Based on the AgileRL MADDPG tutorial for PettingZoo.
This version loads AgileRL hyperparameters and WandB config from YAML files.
"""

import os
import sys
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


def load_yaml_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def save_config_as_wandb_artifact(run, config_path, artifact_name):
    """Save a configuration file as a WandB artifact."""
    artifact = wandb.Artifact(artifact_name, type="config")
    artifact.add_file(config_path)
    run.log_artifact(artifact)
    print(f"Saved {config_path} as WandB artifact: {artifact_name}")


def evaluate_checkpointed_models(run_id, num_episodes=5):
    """Find all checkpointed models with _0_ pattern and evaluate them."""
    import re
    import glob
    import subprocess
    
    # Find all checkpoint files with _0_ pattern
    checkpoint_patterns = [
        f"wandb/latest-run/_0_*.pt",
        f"wandb/run-*-{run_id}/_0_*.pt"
    ]
    
    checkpoint_files = []
    for pattern in checkpoint_patterns:
        checkpoint_files.extend(glob.glob(pattern))
    
    # Remove duplicates
    seen = set()
    checkpoint_files = [f for f in checkpoint_files if f not in seen and not seen.add(f)]
    
    if not checkpoint_files:
        print(f"No checkpoint files found")
        return
    
    # Sort by step number
    def extract_step(filename):
        match = re.search(r'_0_(\d+)\.pt$', filename)
        return int(match.group(1)) if match else 0
    
    checkpoint_files.sort(key=extract_step)
    
    print(f"\n{'='*70}")
    print(f"Found {len(checkpoint_files)} checkpoint files to evaluate")
    print(f"{'='*70}\n")
    
    # Path to the evaluation script
    eval_script_path = "src/elendil_env/test/agilerl_maddpg_eval.py"
    
    # Evaluate each checkpoint
    for checkpoint_file in checkpoint_files:
        model_id = os.path.basename(checkpoint_file).replace('.pt', '')
        
        print(f"\n{'='*70}")
        print(f"Evaluating checkpoint: {model_id}")
        print(f"{'='*70}")
        
        try:
            result = subprocess.run(
                [sys.executable, eval_script_path, model_id, "--num_episodes", str(num_episodes)],
                cwd=os.getcwd(),
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully evaluated {model_id}")
            else:
                print(f"⚠️  Evaluation returned code {result.returncode}")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n{'='*70}")
    print(f"Finished evaluating all checkpoints")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # Load training configuration
    train_config = load_yaml_config("configs/train_configs/agilerl_ippo_test_run.yaml")
    
    # Extract WandB configuration
    wandb_config = train_config["wandb"]
    
    # Extract hyperparameters
    hp = train_config["hyperparameters"]
    
    # Set device
    device = torch.device(hp.get("device", "cpu"))
    print(f"Using device: {device}")

    # Load configuration files
    env_config_path = "configs/env_configs/medium_env_obstacles.yaml"
    env_config = load_yaml_config(env_config_path)
    
    # Load agent configs
    agent_configs = []
    ground_agent_config_path = "configs/agent_configs/ground_agent.yaml"
    ground_config = load_yaml_config(ground_agent_config_path)
    agent_configs.append(ground_config)
    
    air_agent_config_path = "configs/agent_configs/air_observer_agent.yaml"
    air_config = load_yaml_config(air_agent_config_path)
    agent_configs.append(air_config)
    
    target_config_path = "configs/target_configs/target_config.yaml"
    target_config = load_yaml_config(target_config_path)

    # WandB initialization
    run_name = f"{wandb_config['name_prefix']}_{time.strftime('%Y%m%d-%H%M%S')}"
    
    run = wandb.init(
        project=wandb_config["project"],
        name=run_name,
        tags=wandb_config["tags"],
        notes=wandb_config["notes"],
        config={
            "env_name": hp["env_name"],
            "total_timesteps": hp["total_timesteps"],
            "num_envs": hp["num_envs"],
            "env_config": {**env_config},
            "agent_config": agent_configs[0],
            "target_config": {**target_config},
            "algorithm": hp["algorithm"],
            **hp  # Include all hyperparameters in config
        },
        save_code=True,
    )
    
    # Save all config files as WandB artifacts
    print("\nSaving configuration files as WandB artifacts...")
    save_config_as_wandb_artifact(run, "configs/train_configs/agilerl_ippo_test_run.yaml", "train_config")
    save_config_as_wandb_artifact(run, env_config_path, "env_config")
    save_config_as_wandb_artifact(run, ground_agent_config_path, "ground_agent_config")
    save_config_as_wandb_artifact(run, air_agent_config_path, "air_observer_agent_config")
    save_config_as_wandb_artifact(run, target_config_path, "target_config")
    print("All configuration files saved as artifacts.\n")

    # Initialize hyperparameters dict
    INIT_HP = {
        "POPULATION_SIZE": hp["population_size"],
        "ALGO": hp["algorithm"],
        "CHANNELS_LAST": False,
        "BATCH_SIZE": hp["batch_size"],
        "O_U_NOISE": hp["o_u_noise"],
        "EXPL_NOISE": hp["expl_noise"],
        "MEAN_NOISE": hp["mean_noise"],
        "THETA": hp["theta"],
        "DT": hp["dt"],
        "LR_ACTOR": hp["lr_actor"],
        "LR_CRITIC": hp["lr_critic"],
        "GAMMA": hp["gamma"],
        "MEMORY_SIZE": hp["memory_size"],
        "LEARN_STEP": hp["learn_step"],
        "TAU": hp["tau"],
    }

    num_envs = hp["num_envs"]
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

    # Create mutations and tournament selection
    # Note: For population_size=1, mutations may not be needed
    mutation_config = hp.get("mutation", {})
    mutations = Mutations(
        no_mutation=mutation_config.get("no_mutation", 0.2),
        architecture=mutation_config.get("architecture", 0.2),
        new_layer_prob=mutation_config.get("new_layer_prob", 0.2),
        parameters=mutation_config.get("parameters", 0.2),
        activation=mutation_config.get("activation", 0),
        rl_hp=mutation_config.get("rl_hp", 0.2),
        mutation_sd=mutation_config.get("mutation_sd", 0.1),
        mutate_elite=mutation_config.get("mutate_elite", True),
        device=device,
        rand_seed=mutation_config.get("rand_seed", 1),
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
        net_config= None,  # Let AgileRL auto-configure for Box spaces
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

    # Use the provided training function
    print(f"Starting training...")
    
    # Calculate checkpoint interval if not specified
    checkpoint_interval = hp.get("checkpoint_interval")
    if checkpoint_interval is None:
        checkpoint_interval = int(hp["total_timesteps"] / 10)
    
    try:
        trained_pop, pop_fitnesses = train_multi_agent_off_policy(
            env=env,
            env_name="elendil_gridworld",
            algo="MADDPG",
            pop=pop,
            memory=memory,
            INIT_HP=INIT_HP,
            max_steps=hp["total_timesteps"],
            evo_steps=hp["evo_steps"],
            eval_steps=None,
            eval_loop=hp["eval_loop"],
            learning_delay=hp["learning_delay"],
            target=None,
            tournament=tournament,
            mutation=mutations,
            wb=True,
            checkpoint=checkpoint_interval,
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
        # Evaluate all checkpointed models and upload videos to WandB
        if 'run' in locals():
            run_id = run.id
            
            # Resume the run in case AgileRL closed it
            try:
                api = wandb.Api()
                resumed_run = api.run(f"{wandb_config['project']}/{run_id}")
                print(f"\n{'='*70}")
                print(f"Resuming WandB run: {run_id}")
                print(f"{'='*70}")
                
                # Re-initialize wandb with the same run
                wandb.init(
                    project=wandb_config["project"],
                    id=run_id,
                    resume="allow"
                )
            except Exception as e:
                print(f"Could not resume run: {e}")
                # If resume fails, just continue with evaluation
            
            print(f"\n{'='*70}")
            print(f"Starting evaluation of all checkpointed models")
            print(f"{'='*70}")
            evaluate_checkpointed_models(run_id, num_episodes=5)
        
        # Finish WandB run
        wandb.finish()
        print("Training completed and logged to WandB")

    env.close()

