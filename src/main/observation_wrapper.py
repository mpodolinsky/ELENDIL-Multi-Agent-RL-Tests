"""
Observation and Action wrapper to flatten nested spaces for Stable Baselines3 compatibility.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Union


class FlattenObservationWrapper(gym.ObservationWrapper):
    """
    Flattens nested Dict observation spaces into a single Box space.
    This is required for Stable Baselines3 compatibility.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._setup_flattened_space()
    
    def _setup_flattened_space(self):
        """Setup the flattened observation space."""
        if not isinstance(self.env.observation_space, spaces.Dict):
            # If not a Dict, no flattening needed
            self.observation_space = self.env.observation_space
            return
        
        # Calculate total flattened size
        total_size = 0
        self._flatten_info = {}
        
        for agent_id, agent_obs_space in self.env.observation_space.spaces.items():
            if isinstance(agent_obs_space, spaces.Dict):
                # Flatten nested Dict
                agent_size = 0
                agent_flatten_info = {}
                
                for obs_key, obs_space in agent_obs_space.spaces.items():
                    if isinstance(obs_space, spaces.Box):
                        space_size = np.prod(obs_space.shape)
                        agent_flatten_info[obs_key] = {
                            'start': total_size + agent_size,
                            'end': total_size + agent_size + space_size,
                            'shape': obs_space.shape
                        }
                        agent_size += space_size
                    else:
                        raise ValueError(f"Unsupported observation space type: {type(obs_space)}")
                
                self._flatten_info[agent_id] = {
                    'start': total_size,
                    'end': total_size + agent_size,
                    'components': agent_flatten_info
                }
                total_size += agent_size
            else:
                # Direct Box space
                space_size = np.prod(agent_obs_space.shape)
                self._flatten_info[agent_id] = {
                    'start': total_size,
                    'end': total_size + space_size,
                    'shape': agent_obs_space.shape
                }
                total_size += space_size
        
        # Create flattened Box space
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(total_size,), 
            dtype=np.float32
        )
    
    def observation(self, obs: Dict[str, Any]) -> np.ndarray:
        """Flatten the nested observation dictionary."""
        flattened = np.zeros(self.observation_space.shape[0], dtype=np.float32)
        
        for agent_id, agent_obs in obs.items():
            if agent_id not in self._flatten_info:
                continue
                
            agent_info = self._flatten_info[agent_id]
            
            if 'components' in agent_info:
                # Nested Dict case
                for obs_key, obs_value in agent_obs.items():
                    if obs_key in agent_info['components']:
                        comp_info = agent_info['components'][obs_key]
                        start_idx = agent_info['start'] + comp_info['start'] - agent_info['start']
                        end_idx = start_idx + comp_info['end'] - comp_info['start']
                        flattened[start_idx:end_idx] = np.array(obs_value).flatten()
            else:
                # Direct Box case
                start_idx = agent_info['start']
                end_idx = agent_info['end']
                flattened[start_idx:end_idx] = np.array(agent_obs).flatten()
        
        return flattened


class FlattenActionWrapper(gym.ActionWrapper):
    """
    Flattens nested Dict action spaces for Stable Baselines3 compatibility.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._setup_flattened_action_space()
    
    def _setup_flattened_action_space(self):
        """Setup the flattened action space."""
        if not isinstance(self.env.action_space, spaces.Dict):
            # If not a Dict, no flattening needed
            self.action_space = self.env.action_space
            return
        
        # For multi-agent environments, we'll use the first agent's action space
        # This assumes all agents have the same action space
        first_agent_id = list(self.env.action_space.spaces.keys())[0]
        first_agent_action_space = self.env.action_space.spaces[first_agent_id]
        
        # Store the agent ID for action mapping
        self._agent_id = first_agent_id
        
        # Use the first agent's action space directly
        self.action_space = first_agent_action_space
    
    def action(self, action: Union[int, np.ndarray]) -> Dict[str, Any]:
        """Convert flattened action back to multi-agent format."""
        return {self._agent_id: action}


class FlattenMultiAgentWrapper(gym.Wrapper):
    """
    Combined wrapper that flattens both observation and action spaces for multi-agent environments.
    """
    
    def __init__(self, env):
        super().__init__(env)
        # First flatten observations
        self.env = FlattenObservationWrapper(env)
        # Then flatten actions
        self.env = FlattenActionWrapper(self.env)
