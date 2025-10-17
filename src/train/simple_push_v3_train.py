''' First test of HeMAC, AgileRl, and wandb pipeline.
This early file includes both training and testing'''

# train_ippo_hemac_wandb.py
import os
import time
import numpy as np
from tqdm import tqdm
import torch
import wandb
import yaml
from gymnasium import spaces

from mpe2 import simple_push_v3,simple_adversary_v3
from agilerl.algorithms import IPPO
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

# -----------------------
# W&B Setup
# -----------------------
with open("ippo_default_config.yaml", "r") as f:
    config = yaml.safe_load(f)

wandb.init(
    project="mpe2-simple-adversary-v3",
    name=f"ippo_{time.strftime('%Y%m%d-%H%M%S')}",
    config=config,
    monitor_gym=True,
)

# -----------------------
# Environment Setup
# -----------------------
NUM_ENVS = wandb.config.num_envs

def make_env():
    return simple_adversary_v3.parallel_env(
        render_mode=None)

# We can add drone_config to determine additional parameters of the drone

env = AsyncPettingZooVecEnv([make_env for _ in range(NUM_ENVS)])
env.reset()

observation_spaces = [env.single_observation_space(agent) for agent in env.agents]
action_spaces = [env.single_action_space(agent) for agent in env.agents]
agent_ids = list(env.agents)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------
# Initialize IPPO Agent
# -----------------------
agent = IPPO(
    observation_spaces=observation_spaces,
    action_spaces=action_spaces,
    agent_ids=agent_ids,
    device=device,
    batch_size=wandb.config.batch_size,
    lr=wandb.config.lr,
    learn_step=wandb.config.learn_step,
    gamma=wandb.config.gamma,
    gae_lambda=wandb.config.gae_lambda,
    action_std_init=wandb.config.action_std_init,
    clip_coef=wandb.config.clip_coef,
    ent_coef=wandb.config.ent_coef,
    vf_coef=wandb.config.vf_coef,
    max_grad_norm=wandb.config.max_grad_norm,
    update_epochs=wandb.config.update_epochs,
    net_config=wandb.config.net_config
)

print("Shared agent groups:", agent.shared_agent_ids)
print("Test")

# -----------------------
# Training Loop
# -----------------------
max_steps = wandb.config.max_steps
pbar = tqdm(total=max_steps, desc="Training progress")
total_steps = 0
episode_rewards = []

os.makedirs("wandb/latest-run/checkpoints", exist_ok=True)

while total_steps < max_steps:
    obs, info = env.reset()
    scores = np.zeros((NUM_ENVS, len(agent.shared_agent_ids)))
    completed_scores = []
    done = {agent_id: np.zeros(NUM_ENVS) for agent_id in agent.agent_ids}

    while True:
        # Initialize rollout containers
        states, actions, log_probs, entropies = {}, {}, {}, {}
        rewards, dones, values = {}, {}, {}
        for a_id in agent.agent_ids:
            states[a_id], actions[a_id], log_probs[a_id] = [], [], []
            entropies[a_id], rewards[a_id], dones[a_id], values[a_id] = [], [], [], []

        # Rollout collection
        rollout_steps = (agent.learn_step + NUM_ENVS - 1) // NUM_ENVS
        for _ in range(rollout_steps):
            action, log_prob, entropy, value = agent.get_action(obs=obs, infos=info)

            # Clip to action space
            clipped_action = {}
            for agent_id, agent_action in action.items():
                network_id = (
                    agent_id
                    if agent_id in agent.actors.keys()
                    else agent.get_group_id(agent_id)
                )
                agent_space = agent.possible_action_spaces[agent_id]
                if isinstance(agent_space, spaces.Box):
                    if agent.actors[network_id].squash_output:
                        clipped_agent_action = agent.actors[
                            network_id
                        ].scale_action(agent_action)
                    else:
                        clipped_agent_action = np.clip(
                            agent_action, agent_space.low, agent_space.high
                        )
                else:
                    clipped_agent_action = agent_action

                clipped_action[agent_id] = clipped_agent_action

            # Step the environment
            next_obs, reward, termination, truncation, info = env.step(clipped_action)

            # Handle rewards and episode scoring
            agent_rewards = np.array(list(reward.values())).T
            agent_rewards = np.nan_to_num(agent_rewards)
            scores += np.sum(agent_rewards, axis=-1)[:, np.newaxis]
            total_steps += NUM_ENVS

            # Store transitions
            for a_id in obs:
                states[a_id].append(obs[a_id])
                actions[a_id].append(action[a_id])
                log_probs[a_id].append(log_prob[a_id])
                entropies[a_id].append(entropy[a_id])
                rewards[a_id].append(reward[a_id])
                values[a_id].append(value[a_id])
                dones[a_id].append(done[a_id])

            # Compute done mask
            next_done = {}
            for a_id in termination:
                terminated = termination[a_id]
                truncated = truncation[a_id]
                mask = ~(np.isnan(terminated) | np.isnan(truncated))
                result = np.full_like(mask, np.nan, dtype=float)
                result[mask] = np.logical_or(terminated[mask], truncated[mask])
                next_done[a_id] = result

            obs = next_obs
            done = next_done

            # Track completed episodes
            for idx, agent_dones in enumerate(zip(*next_done.values())):
                if all(agent_dones):
                    completed_scores.append(scores[idx].tolist())
                    episode_rewards.append(np.sum(scores[idx]))
                    scores[idx].fill(0)
                    done = {a_id: np.zeros(NUM_ENVS) for a_id in agent.agent_ids}

            if total_steps >= max_steps:
                break

        # --- PPO update ---
        experiences = (states, actions, log_probs, rewards, dones, values, next_obs, next_done)
        loss = agent.learn(experiences)

        # --- Logging & Checkpointing ---
        SAVE_INTERVAL = 10  # save a checkpoint every 10 PPO updates

        # Each rollout collects roughly learn_step * NUM_ENVS total interactions
        if total_steps % (rollout_steps * NUM_ENVS) == 0 or total_steps >= max_steps:
            # --- Compute rolling average reward ---
            if len(episode_rewards) >= 10:
                avg_score = np.mean(episode_rewards[-10:]) # TODO Does this actually represent the number we want?
            else:
                avg_score = np.mean(episode_rewards) if episode_rewards else 0.0

            # --- Save checkpoint every N rollouts ---
            updates_completed = total_steps // (rollout_steps * NUM_ENVS)
            if updates_completed % SAVE_INTERVAL == 0 or total_steps >= max_steps:
                checkpoint_path = f"wandb/latest-run/checkpoints/ippo_step{total_steps}.pt"
                os.makedirs(os.path.dirname(checkpoint_path), exist_ok=True)
                agent.save_checkpoint(checkpoint_path)

                # Log model as W&B artifact
                artifact = wandb.Artifact("model", type="model")
                artifact.add_file(checkpoint_path)
                wandb.log_artifact(artifact)

            # --- Log metrics to W&B ---
            wandb.log(
                {
                    **loss,                     # PPO losses returned by agent.learn()
                    "mean_reward_10ep": avg_score,
                    "total_steps": total_steps, # true total environment steps
                },
                step=total_steps,                # ensure x-axis = total env steps
            )

            # --- Update progress bar ---
            pbar.set_description(f"Avg score: {avg_score:.2f}")
            pbar.update(rollout_steps * NUM_ENVS)


        if total_steps >= max_steps:
            break

env.close()
wandb.finish()
print("Training completed.")