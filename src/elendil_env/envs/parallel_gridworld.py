"""
Parallel version of GridWorldEnvMultiAgent using PettingZoo's aec_to_parallel converter.
"""
import sys
import os

# Add HA-SPO2V-Env to Python path to fix import issues
ha_spo2v_path = "/mnt/data/Documents/Project_M/HA-SPO2V-Env"
if ha_spo2v_path not in sys.path:
    sys.path.insert(0, ha_spo2v_path)

from pettingzoo.utils import aec_to_parallel
from gymnasium_env.envs.grid_world_multi_agent import GridWorldEnvMultiAgent


def parallel_env(**kwargs):
    """
    Create a parallel version of GridWorldEnvMultiAgent.
    
    This wraps the AEC environment and converts it to parallel execution,
    where all agents act simultaneously.
    
    Args:
        **kwargs: All arguments passed to GridWorldEnvMultiAgent
        
    Returns:
        ParallelEnv: Parallel version of the environment
    """
    aec_env = GridWorldEnvMultiAgent(**kwargs)
    return aec_to_parallel(aec_env)


# Convenience function with default parameters
def env(**kwargs):
    """Alias for parallel_env for PettingZoo compatibility."""
    return parallel_env(**kwargs)

