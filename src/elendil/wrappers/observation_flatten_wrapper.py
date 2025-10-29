"""Observation flatten wrapper for AgileRL compatibility."""

import numpy as np
from gymnasium import Wrapper
from gymnasium.spaces import Box, Dict, Discrete, MultiDiscrete
from typing import Dict as TypingDict, Any

from pettingzoo.utils.env import ParallelEnv


class ObservationFlattenWrapper(ParallelEnv):
    """Wrapper that flattens Dict observation spaces to Box for AgileRL compatibility."""
    
    def __init__(self, env: ParallelEnv):
        super().__init__()
        self.env = env
        self.metadata = env.metadata
        self.agents = env.agents
        self.possible_agents = env.possible_agents
        # Forward render_mode attribute for compatibility
        self.render_mode = getattr(env, 'render_mode', None)
        
        # Calculate flattened observation space dimensions
        self._flattened_dims = {}
        for agent_id in self.agents:
            obs_space = self.env.observation_space(agent_id)
            self._flattened_dims[agent_id] = self._calculate_flattened_dim(obs_space)
        
        # Create Box observation spaces for each agent
        self._observation_spaces = {
            agent_id: Box(
                low=-np.inf,
                high=np.inf,
                shape=(dim,),
                dtype=np.float32
            )
            for agent_id, dim in self._flattened_dims.items()
        }
        
        # Keep action spaces the same
        self._action_spaces = {
            agent_id: self.env.action_space(agent_id)
            for agent_id in self.agents
        }
    
    def _calculate_flattened_dim(self, obs_space) -> int:
        """Calculate the flattened dimension of an observation space."""
        if isinstance(obs_space, Box):
            return int(np.prod(obs_space.shape))
        elif isinstance(obs_space, Discrete):
            return 1
        elif isinstance(obs_space, MultiDiscrete):
            return int(np.sum(obs_space.nvec))
        elif isinstance(obs_space, Dict):
            total_dim = 0
            for key, space in obs_space.spaces.items():
                if isinstance(space, Box):
                    total_dim += int(np.prod(space.shape))
                elif isinstance(space, Discrete):
                    total_dim += 1
                elif isinstance(space, MultiDiscrete):
                    total_dim += int(np.sum(space.nvec))
            return total_dim
        else:
            raise ValueError(f"Unsupported observation space type: {type(obs_space)}")
    
    def _flatten_observation(self, obs: TypingDict[str, Any]) -> np.ndarray:
        """Flatten a Dict observation into a 1D array."""
        flat_list = []
        for key in sorted(obs.keys()):
            value = obs[key]
            if isinstance(value, np.ndarray):
                flat_list.append(value.flatten())
            else:
                flat_list.append(np.array([value]))
        return np.concatenate(flat_list).astype(np.float32)
    
    def observation_space(self, agent):
        """Return the Box observation space for the agent."""
        return self._observation_spaces[agent]
    
    def action_space(self, agent):
        """Return the action space for the agent."""
        return self._action_spaces[agent]
    
    def reset(self, seed=None, options=None):
        """Reset the environment and return flattened observations."""
        obs, info = self.env.reset(seed=seed, options=options)
        
        # Flatten observations for all agents
        flattened_obs = {}
        for agent_id, observation in obs.items():
            flattened_obs[agent_id] = self._flatten_observation(observation)
        
        return flattened_obs, info
    
    def step(self, actions):
        """Step the environment and return flattened observations."""
        next_obs, rewards, terminations, truncations, infos = self.env.step(actions)
        
        # Flatten next observations for all agents
        flattened_next_obs = {}
        for agent_id, observation in next_obs.items():
            flattened_next_obs[agent_id] = self._flatten_observation(observation)
        
        return flattened_next_obs, rewards, terminations, truncations, infos
    
    def render(self):
        """Render the environment."""
        return self.env.render()
    
    def close(self):
        """Close the environment."""
        return self.env.close()

