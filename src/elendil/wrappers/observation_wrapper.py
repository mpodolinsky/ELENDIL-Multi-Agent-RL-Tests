"""
Observation and Action wrapper to flatten nested spaces for Stable Baselines3 compatibility.
Also converts PettingZoo AEC environments to Gymnasium-compatible format.
"""
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from typing import Dict, Any, Union
from pettingzoo.utils.env import AECEnv


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


class AECToGymnasiumWrapper(gym.Env):
    """
    Wraps a PettingZoo AEC environment to make it compatible with Gymnasium/Stable Baselines3.
    """
    
    def __init__(self, aec_env):
        super().__init__()
        self.aec_env = aec_env
        
        # Get the observation and action spaces from the AEC env
        # Convert Python dict to gymnasium.spaces.Dict if necessary
        if hasattr(aec_env, 'observation_space'):
            obs_space = aec_env.observation_space
            if isinstance(obs_space, dict) and not isinstance(obs_space, spaces.Dict):
                # Convert regular dict to gymnasium.spaces.Dict
                self.observation_space = spaces.Dict(obs_space)
            else:
                self.observation_space = obs_space
        
        if hasattr(aec_env, 'action_space'):
            act_space = aec_env.action_space
            if isinstance(act_space, dict) and not isinstance(act_space, spaces.Dict):
                # Convert regular dict to gymnasium.spaces.Dict
                self.action_space = spaces.Dict(act_space)
            else:
                self.action_space = act_space
        
        # Preserve render_mode for VecVideoRecorder compatibility
        if hasattr(aec_env, 'render_mode'):
            self.render_mode = aec_env.render_mode
        else:
            self.render_mode = None
        
        # Also preserve metadata if available
        if hasattr(aec_env, 'metadata'):
            self.metadata = aec_env.metadata
    
    def reset(self, seed=None, options=None):
        """Reset the environment."""
        if seed is not None:
            # PettingZoo AEC envs might not support seed in reset
            try:
                obs = self.aec_env.reset(seed=seed)
            except TypeError:
                obs = self.aec_env.reset()
        else:
            obs = self.aec_env.reset()
        
        info = {}
        return obs, info
    
    def step(self, action):
        """Step the environment."""
        result = self.aec_env.step(action)
        
        # Handle different return formats
        if len(result) == 5:
            obs, reward, terminated, truncated, info = result
        elif len(result) == 4:
            obs, reward, done, info = result
            terminated = done
            truncated = False
        else:
            raise ValueError(f"Unexpected step return format: {len(result)} values")
        
        return obs, reward, terminated, truncated, info
    
    def render(self):
        """Render the environment."""
        if hasattr(self.aec_env, 'render'):
            return self.aec_env.render()
    
    def close(self):
        """Close the environment."""
        if hasattr(self.aec_env, 'close'):
            self.aec_env.close()


class FlattenMultiAgentWrapper(gym.Wrapper):
    """
    Combined wrapper that:
    1. Converts AEC to Gymnasium (if needed)
    2. Flattens observation spaces
    3. Flattens action spaces
    """
    
    def __init__(self, env):
        # First, check if we need to wrap as Gymnasium env
        if isinstance(env, AECEnv):
            env = AECToGymnasiumWrapper(env)
        
        super().__init__(env)
        # Then flatten observations
        self.env = FlattenObservationWrapper(env)
        # Then flatten actions
        self.env = FlattenActionWrapper(self.env)
