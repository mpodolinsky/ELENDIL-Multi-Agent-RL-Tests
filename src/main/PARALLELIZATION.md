# Environment Parallelization Guide

This guide explains how to parallelize your AEC (Agent Environment Cycle) environment for faster training.

## Current Implementation

The main script (`custom_env_main.py`) uses **SubprocVecEnv** for parallelization, which runs multiple environment instances in separate processes.

**Important:** The script uses `if __name__ == '__main__':` to protect the main execution code. This is **required** for multiprocessing on Python 3.12+ to prevent child processes from re-executing the training code.

### Configuration

```python
config = {
    "num_envs": 8,  # Number of parallel environments
    # ... other config
}
```

- Set `num_envs = 1` for single environment (no parallelization, useful for debugging)
- Set `num_envs > 1` for parallel training (recommended: 4-16 depending on your CPU)

## Parallelization Options

### 1. SubprocVecEnv (Current - Recommended)

**How it works:** Creates multiple processes, each running an independent environment instance.

**Pros:**
- True parallelism (uses multiple CPU cores)
- Fastest for CPU-intensive environments
- Isolated processes (crashes don't affect others)

**Cons:**
- Higher memory usage (each process has its own memory)
- Slower startup time
- May have issues with GPU environments (use DummyVecEnv instead)

**When to use:** Most scenarios, especially CPU-bound environments

```python
env = SubprocVecEnv([make_env for _ in range(num_envs)])
```

### 2. DummyVecEnv (Alternative)

**How it works:** Runs all environments sequentially in a single process.

**Pros:**
- Lower memory usage
- Easier debugging
- Works well with GPU environments

**Cons:**
- No true parallelism (sequential execution)
- Slower than SubprocVecEnv

**When to use:** 
- Debugging
- GPU-heavy environments
- Memory-constrained systems

```python
env = DummyVecEnv([make_env for _ in range(num_envs)])
```

## AEC Environment Specifics

Your `GridWorldEnvMultiAgent` is an AEC environment, which means agents take turns acting. The current implementation:

1. **Flattens** the multi-agent structure using `FlattenMultiAgentWrapper`
2. **Converts** the Dict observation/action spaces to single spaces
3. **Vectorizes** using SubprocVecEnv for parallel execution

This approach treats the multi-agent environment as a single-agent environment for Stable Baselines3 compatibility.

## Performance Tips

### Optimal num_envs

- **CPU cores:** Set `num_envs` ≤ number of CPU cores
- **Memory:** Each environment instance requires memory; reduce if memory-limited
- **Recommended:** Start with 4-8 environments

### Training Speed

With `num_envs = 8`, you'll collect experiences ~8x faster compared to single environment:
- Single env: 250,000 timesteps takes ~X minutes
- 8 parallel envs: 250,000 timesteps takes ~X/8 minutes

### Monitoring

Watch your CPU usage:
```bash
htop  # Linux/Mac
# or
nvidia-smi  # For GPU usage
```

If CPU usage is < 100% per core, you can increase `num_envs`.

## Troubleshooting

### RuntimeError: "An attempt has been made to start a new process..."

**Cause:** Your script is missing the `if __name__ == '__main__':` guard around the main execution code.

**Solution:** Wrap all training code (WandB init, environment creation, model training) in:
```python
if __name__ == '__main__':
    # Your training code here
    run = wandb.init(...)
    env = SubprocVecEnv(...)
    # etc.
```

This prevents child processes from re-executing the training code.

### Multiple WandB runs created

**Symptom:** You see 8+ WandB runs instead of 1.

**Cause:** Same as above - missing `if __name__ == '__main__':` guard.

**Solution:** Apply the fix above. Each subprocess was initializing its own WandB run.

### "Too many open files" error
Increase system limits:
```bash
ulimit -n 4096
```

### Out of memory
Reduce `num_envs` or reduce environment complexity.

### Slow startup
SubprocVecEnv takes time to spawn processes. This is normal.

### Video recording issues
Only the first environment (id=0) records videos to avoid conflicts. This is handled automatically.

## Alternative: PettingZoo Vectorization

For native PettingZoo environments, you can also use:

```python
from agilerl.vector.pz_async_vec_env import AsyncPettingZooVecEnv

env = AsyncPettingZooVecEnv([make_env for _ in range(num_envs)])
```

This is used in `simple_push_v3_train.py` and is specifically designed for multi-agent environments.

