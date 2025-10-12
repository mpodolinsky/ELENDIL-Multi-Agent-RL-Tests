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

from hemac import HeMAC_v0
from agilerl.algorithms import IPPO
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

# -----------------------
# W&B Setup
# -----------------------
with open("ippo_default_config.yaml", "r") as f:
    config = yaml.safe_load(f)

wandb.init(
    project="HeMAC-IPPO",
    name=f"ippo_{time.strftime('%Y%m%d-%H%M%S')}",
    config=config
)

# -----------------------
# Environment Setup
# -----------------------
NUM_ENVS = wandb.config.num_envs

def make_env():
    return HeMAC_v0.parallel_env(
        render_mode=None,
        n_drones=3,
        n_observers=1,
        area_size=(500, 500),
        max_cycles=600
    )

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
    batch_size=128,
    rollout_steps=wandb.config.rollout_steps,
    learn_step=wandb.config.learn_step,
    gamma=wandb.config.gamma,
    gae_lambda=wandb.config.gae_lambda,
    clip_coef=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    learning_rate=wandb.config.lr,
    max_grad_norm=0.5,
)
print("Shared agent groups:", agent.shared_agent_ids)

# -----------------------
# Training Loop
# -----------------------
MAX_STEPS = 200_000
CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

obs, info = env.reset()
steps = 0
pbar = tqdm(total=MAX_STEPS)

while steps < MAX_STEPS:
    agent.rollout_steps_collected = 0
    group_rewards = {gid: [] for gid in agent.shared_agent_ids}

    while agent.rollout_steps_collected < agent.learn_step:
        actions, log_prob, entropy, values = agent.get_action(obs=obs, infos=info)
        next_obs, rewards, terminations, truncations, infos = env.step(actions)

        agent.remember(obs, actions, rewards, terminations, truncations,
                       log_prob, values, infos)
        obs = next_obs
        steps += NUM_ENVS
        pbar.update(NUM_ENVS)

        # Log rewards by group
        for gid in agent.shared_agent_ids:
            group_rewards[gid].append(sum(r for aid, r in rewards.items()
                                          if agent.get_group_id(aid) == gid))

    # Learn from rollout
    loss_dict = agent.learn()

    # Aggregate and log metrics to W&B
    mean_rewards = {gid: np.mean(vals) for gid, vals in group_rewards.items()}
    wandb.log({
        "steps": steps,
        **{f"reward/{gid}": val for gid, val in mean_rewards.items()},
        **{f"loss/{k}": v for k, v in loss_dict.items()},
    })

    # Save checkpoints periodically
    if steps % 50000 < NUM_ENVS:
        ckpt_path = f"{CHECKPOINT_DIR}/ippo_{steps//1000}k.pt"
        agent.save_checkpoint(ckpt_path)
        wandb.save(ckpt_path)

env.close()
wandb.finish()
print("✅ Training complete.")
