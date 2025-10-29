# Multiprocessing Fix: `if __name__ == '__main__':`

## The Problem

When using `SubprocVecEnv` for parallel environment execution, you encountered this error:

```
RuntimeError: 
    An attempt has been made to start a new process before the
    current process has finished its bootstrapping phase.
```

And multiple WandB runs were created instead of one.

## Why This Happened

Python's multiprocessing with the `'spawn'` or `'forkserver'` start method (default in Python 3.12+) works by:

1. **Re-importing your entire script** in each child process
2. **Re-executing all top-level code** in each child process

Without protection, this creates a recursive loop:
- Main process starts
- Main process tries to spawn 8 child processes
- Each child process re-runs your script from the top
- Each child tries to spawn 8 more child processes
- **9+ WandB runs get created** (1 main + 8 children)
- Script crashes with RuntimeError

## The Solution

Wrap all main execution code in `if __name__ == '__main__':` block:

```python
# ✅ GOOD: Configuration and functions at module level (can be imported)
config = {...}
agent_configs = [...]

def make_env():
    # Environment creation
    return env

def record_video_trigger(step):
    # Helper function
    return condition

# ✅ GOOD: Main execution protected
if __name__ == '__main__':
    # This ONLY runs in the main process
    run = wandb.init(...)
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    model = PPO(...)
    model.learn(...)
```

### What Goes Where

**Outside `if __name__ == '__main__':`** (module level):
- Imports
- Configuration loading (YAML files, etc.)
- Function and class definitions
- Constants

**Inside `if __name__ == '__main__':`** (main execution):
- WandB initialization
- Environment creation (SubprocVecEnv)
- Model creation and training
- Saving and uploading results

## How It Works

When child processes spawn:
1. They import your script
2. They load configurations and function definitions
3. They **skip** the `if __name__ == '__main__':` block
4. They wait for instructions from the main process

The main process:
1. Imports and loads everything
2. **Executes** the `if __name__ == '__main__':` block
3. Creates SubprocVecEnv, which spawns children
4. Coordinates training across all processes

## Verification

After the fix:
- ✅ Only **1 WandB run** is created
- ✅ Multiple Python processes run (1 main + 8 workers)
- ✅ No RuntimeError
- ✅ Training proceeds normally

You can verify multiple processes are running:
```bash
ps aux | grep python | grep custom_env_main
```

You should see multiple entries (one main process + child processes).

## Why This Is Required

This is not specific to your code - it's a **fundamental requirement** of Python multiprocessing when using `'spawn'` or `'forkserver'` start methods.

- **Linux (Python 3.12+):** Uses `'forkserver'` by default → requires `if __name__ == '__main__':`
- **macOS/Windows:** Always use `'spawn'` → requires `if __name__ == '__main__':`
- **Linux (Python < 3.8):** Uses `'fork'` by default → doesn't strictly require it (but still recommended)

## Best Practice

**Always** use `if __name__ == '__main__':` when:
- Using any multiprocessing (SubprocVecEnv, Process, Pool, etc.)
- Creating scripts that might be imported by other scripts
- Writing production-ready Python code

It's a Python best practice and costs nothing to include.

