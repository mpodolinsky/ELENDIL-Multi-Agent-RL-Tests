# Migration to `make_vec_env` - Summary

## What Changed

Successfully migrated from manual `SubprocVecEnv` creation to using Stable Baselines3's `make_vec_env` utility function.

## Why This Is Better

### Before (Manual SubprocVecEnv):
```python
def make_env():
    env = GridWorldEnvMultiAgent(...)
    env = FlattenMultiAgentWrapper(env)
    env = Monitor(env)
    return env

if __name__ == '__main__':
    run = wandb.init(...)
    env = SubprocVecEnv([make_env for _ in range(num_envs)])
    env = VecVideoRecorder(...)
```

**Problems:**
- `make_env()` used `wandb.config`, which doesn't exist in child processes
- Required manual `if __name__ == '__main__':` guards
- Manual Monitor wrapping
- More boilerplate code
- Easy to make mistakes with multiprocessing

### After (Using make_vec_env):
```python
def make_single_env():
    """Uses module-level config variables, not wandb.config"""
    def _init():
        env = GridWorldEnvMultiAgent(
            size=env_config["size"],  # Uses module-level variable
            max_steps=env_config["max_steps"],
            # ...
        )
        env = FlattenMultiAgentWrapper(env)
        return env  # No Manual Monitor wrapping needed
    return _init

if __name__ == '__main__':
    run = wandb.init(...)
    env = make_vec_env(
        make_single_env(),
        n_envs=num_envs,
        vec_env_cls=SubprocVecEnv if num_envs > 1 else DummyVecEnv,
        monitor_dir=f"wandb/latest-run/monitors/{run.id}",
    )
    env = VecVideoRecorder(...)
```

**Benefits:**
- ✅ `make_vec_env` automatically handles multiprocessing correctly
- ✅ No dependence on `wandb.config` in child processes
- ✅ Automatic Monitor wrapping
- ✅ Cleaner, more maintainable code
- ✅ Handles edge cases automatically
- ✅ Less boilerplate

## Key Differences

### 1. Configuration Access

**Before:**
- Used `wandb.config.env_config["size"]` inside `make_env()`
- Failed because `wandb.config` doesn't exist in child processes

**After:**
- Uses module-level variables: `env_config["size"]`
- Works because these variables are available when the module is imported

### 2. Environment Factory Pattern

**Before:**
```python
def make_env():
    return configured_env
```

**After:**
```python
def make_single_env():
    def _init():
        return configured_env
    return _init
```

`make_vec_env` expects a callable that returns a callable (factory pattern).

### 3. Monitor Wrapping

**Before:**
- Manual: `env = Monitor(env)`

**After:**
- Automatic via `monitor_dir` parameter
- Creates monitoring logs in specified directory

### 4. Vec Env Class Selection

**Before:**
```python
if num_envs > 1:
    env = SubprocVecEnv([...])
else:
    env = DummyVecEnv([...])
```

**After:**
```python
vec_env_cls=SubprocVecEnv if num_envs > 1 else DummyVecEnv
```

Single line, cleaner logic.

## Testing Results

✅ **Only 1 WandB run created** (previously 9)
✅ **Multiple processes spawned correctly** for parallelization
✅ **No multiprocessing errors**
✅ **Training running successfully**

## Best Practices

When using `make_vec_env`:

1. **Environment factory should use module-level variables**, not `wandb.config`
2. **Return a callable that returns a callable** (factory pattern)
3. **Don't wrap with Monitor manually** - let `make_vec_env` handle it
4. **Use `monitor_dir` to save training stats**
5. **Keep `if __name__ == '__main__':` guard** - still good practice

## Migration Checklist

If migrating your own code:

- [ ] Replace manual `SubprocVecEnv`/`DummyVecEnv` with `make_vec_env`
- [ ] Change environment factory to not depend on `wandb.config`
- [ ] Use module-level config variables instead
- [ ] Implement factory pattern (callable returning callable)
- [ ] Remove manual `Monitor` wrapping
- [ ] Add `monitor_dir` parameter to `make_vec_env`
- [ ] Test that only 1 WandB run is created
- [ ] Verify multiprocessing works (check process count)

## References

- [Stable Baselines3 make_vec_env documentation](https://stable-baselines3.readthedocs.io/en/master/guide/vec_envs.html#make-vec-env)
- [Python multiprocessing best practices](https://docs.python.org/3/library/multiprocessing.html#programming-guidelines)

