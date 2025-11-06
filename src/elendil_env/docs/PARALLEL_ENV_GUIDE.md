# Converting AEC to Parallel Environment Guide

## Overview

You have two options for using your environment with Stable Baselines3:

1. **Current Approach**: Using custom wrappers (already working)
2. **Native Parallel Approach**: Using PettingZoo's `aec_to_parallel` + SuperSuit (recommended for multi-agent)

## Option 1: Current Approach (Already Implemented) ✅

**Files:** `src/main/custom_env_main.py` + `src/main/observation_wrapper.py`

This approach:
- Converts AEC → Gymnasium
- Flattens observations and actions
- Uses `make_vec_env` for parallelization
- **Status: Working with `num_envs=1`, ready for `num_envs=8`**

**Pros:**
- Already working
- Clean integration with SB3
- Good for single-agent or centralized control

**Cons:**
- Custom wrappers (more maintenance)
- Treats multi-agent as single-agent

## Option 2: Native Parallel Approach (New)

**Files:** `src/envs/parallel_gridworld.py` + `src/main/parallel_train.py`

### Step 1: Create Parallel Wrapper

```python
# src/envs/parallel_gridworld.py
from pettingzoo.utils import aec_to_parallel
from gymnasium_env.envs.grid_world_multi_agent import GridWorldEnvMultiAgent

def parallel_env(**kwargs):
    """
    Create a parallel version of GridWorldEnvMultiAgent.
    All agents act simultaneously.
    """
    aec_env = GridWorldEnvMultiAgent(**kwargs)
    return aec_to_parallel(aec_env)
```

### Step 2: Use SuperSuit for SB3 Compatibility

```python
# src/main/parallel_train.py
from supersuit import pettingzoo_env_to_vec_env_v1, concat_vec_envs_v1
from stable_baselines3 import PPO
from envs.parallel_gridworld import parallel_env

if __name__ == '__main__':
    # Create parallel environment
    base_env = parallel_env(
        size=15,
        max_steps=500,
        render_mode='rgb_array',
        # ... other params
    )
    
    # Convert to SB3 vectorized format
    vec_env = pettingzoo_env_to_vec_env_v1(base_env)
    
    # Create multiple parallel instances
    vec_env = concat_vec_envs_v1(
        vec_env, 
        num_vec_envs=8,  # Number of parallel environments
        num_cpus=8,
        base_class='subproc'  # Use multiprocessing
    )
    
    # Train with PPO
    model = PPO("MlpPolicy", vec_env, verbose=1)
    model.learn(total_timesteps=250_000)
```

### How Parallel Environment Works

**Reset:**
```python
observations, infos = env.reset()
# observations = {'agent_1': obs1, 'agent_2': obs2, ...}
```

**Step (All agents act simultaneously):**
```python
actions = {'agent_1': action1, 'agent_2': action2, ...}
obs, rewards, terminations, truncations, infos = env.step(actions)
```

### Key Differences

| Feature | AEC (Sequential) | Parallel (Simultaneous) |
|---------|-----------------|-------------------------|
| Agent turns | One at a time | All at once |
| Step input | Single action | Dict of actions |
| Step output | Single obs/reward | Dict of obs/rewards |
| Realism | Turn-based games | Real-time scenarios |
| Performance | Slower | Faster |

## Recommendation

**Use Option 1 (Current)** if:
- You want centralized control
- Single agent or treating multi-agent as single
- Already working, just scale up `num_envs`

**Use Option 2 (Parallel)** if:
- You want true multi-agent learning
- Independent agent policies
- Native PettingZoo compatibility

## Next Steps with Current Approach

Since your current implementation is working, you can:

1. **Scale up parallelization:**
   ```python
   config["num_envs"] = 8  # Change from 1 to 8
   ```

2. **Run training:**
   ```bash
   python src/main/custom_env_main.py
   ```

3. **Monitor performance:**
   - Check CPU usage (`htop`)
   - Watch WandB for training metrics
   - Verify ~8x speedup

## Files Created

- ✅ `src/envs/parallel_gridworld.py` - Parallel environment wrapper
- ✅ `src/main/parallel_train.py` - Training script using SuperSuit
- ✅ `src/test/test_parallel_env.py` - Test script for parallel env
- ✅ This guide

## Troubleshooting

If you get import errors with the external `gymnasium_env` package:
1. Check if the package is properly installed
2. Verify the import paths in the package
3. Consider using the current working approach (Option 1)

The parallel approach requires the external package to have correct internal imports, which may need to be fixed in that package.

