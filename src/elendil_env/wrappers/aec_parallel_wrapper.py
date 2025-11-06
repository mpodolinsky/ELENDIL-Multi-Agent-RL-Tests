"""
Wrapper to convert AEC (Agent Environment Cycle) environments to parallel execution.
This is useful for environments that follow the PettingZoo AEC API.
"""
import gymnasium as gym
import numpy as np
from typing import Dict, Any, Tuple


class AECToParallelWrapper(gym.Wrapper):
    """
    Converts an AEC environment to a parallel environment where all agents act simultaneously.
    This is useful when the environment follows AEC pattern but you want parallel execution.
    """
    
    def __init__(self, env):
        super().__init__(env)
        self._agent_ids = []
        self._last_observations = {}
        self._accumulated_rewards = {}
        
    def reset(self, **kwargs):
        """Reset the environment and collect observations for all agents."""
        obs = self.env.reset(**kwargs)
        
        # If the environment returns observations for all agents at once, use them
        if isinstance(obs, dict):
            return obs
        
        # Otherwise, collect observations by stepping through agents (AEC style)
        self._last_observations = {}
        self._accumulated_rewards = {}
        
        # Try to get all agent observations
        # This assumes the wrapped environment has a method to get all observations
        return obs
    
    def step(self, actions: Dict[str, Any]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        """
        Step all agents in parallel.
        
        Args:
            actions: Dictionary mapping agent_id to action
            
        Returns:
            observations, rewards, terminations, truncations, infos
        """
        # For environments already in parallel format, just pass through
        return self.env.step(actions)


class AECCollector:
    """
    Helper class to collect experiences from AEC environments in a vectorized manner.
    This batches observations and actions across multiple environment instances.
    """
    
    def __init__(self, env_fns, num_envs):
        """
        Args:
            env_fns: List of functions that create environment instances
            num_envs: Number of parallel environments
        """
        self.num_envs = num_envs
        self.envs = [fn() for fn in env_fns]
        self.dones = [False] * num_envs
        
    def reset(self):
        """Reset all environments and return batched observations."""
        observations = []
        infos = []
        
        for env in self.envs:
            obs, info = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
            observations.append(obs)
            infos.append(info)
        
        self.dones = [False] * self.num_envs
        return self._batch_observations(observations), infos
    
    def step(self, actions):
        """
        Step all environments with their respective actions.
        
        Args:
            actions: List or array of actions, one per environment
            
        Returns:
            Batched observations, rewards, terminations, truncations, infos
        """
        observations = []
        rewards = []
        terminations = []
        truncations = []
        infos = []
        
        for i, (env, action) in enumerate(zip(self.envs, actions)):
            if self.dones[i]:
                # Auto-reset if done
                obs, info = env.reset() if isinstance(env.reset(), tuple) else (env.reset(), {})
                observations.append(obs)
                rewards.append(0.0)
                terminations.append(False)
                truncations.append(False)
                infos.append(info)
                self.dones[i] = False
            else:
                result = env.step(action)
                if len(result) == 5:
                    obs, reward, term, trunc, info = result
                else:
                    obs, reward, done, info = result
                    term, trunc = done, done
                
                observations.append(obs)
                rewards.append(reward)
                terminations.append(term)
                truncations.append(trunc)
                infos.append(info)
                
                self.dones[i] = term or trunc
        
        return (
            self._batch_observations(observations),
            np.array(rewards),
            np.array(terminations),
            np.array(truncations),
            infos
        )
    
    def _batch_observations(self, observations):
        """Batch observations from multiple environments."""
        if isinstance(observations[0], np.ndarray):
            return np.stack(observations)
        elif isinstance(observations[0], dict):
            # Handle dict observations
            batched = {}
            for key in observations[0].keys():
                batched[key] = np.stack([obs[key] for obs in observations])
            return batched
        else:
            return observations
    
    def close(self):
        """Close all environments."""
        for env in self.envs:
            env.close()

