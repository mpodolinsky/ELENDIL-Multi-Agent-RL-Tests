"""
Benchmark script to compare different parallelization settings.
This helps determine the optimal num_envs for your system.

Usage:
    python src/main/benchmark_parallel.py
"""

import time
import yaml
from gymnasium_env.envs.grid_world_multi_agent import GridWorldEnvMultiAgent
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from observation_wrapper import FlattenMultiAgentWrapper


def make_env(config):
    """Create a single environment instance."""
    def _init():
        env = GridWorldEnvMultiAgent(
            size=config["size"],
            max_steps=config["max_steps"],
            render_mode=None,  # No rendering for benchmarking
            show_fov_display=False,
            intrinsic=False,
            lambda_fov=config.get("lambda_fov", 0.5),
            show_target_coords=False,
            obstacle_collision_penalty=-0.05,
            no_target=False,
            agents=config["agents"],
            target_config=config["target_config"],
            enable_obstacles=config.get("enable_obstacles", True),
            num_obstacles=config.get("num_obstacles", 3),
            num_visual_obstacles=config.get("num_visual_obstacles", 0),
        )
        env = FlattenMultiAgentWrapper(env)
        env = Monitor(env)
        return env
    return _init


def benchmark(num_envs, timesteps=10000, use_subproc=True):
    """
    Benchmark training with specified number of environments.
    
    Args:
        num_envs: Number of parallel environments
        timesteps: Number of timesteps to train
        use_subproc: If True, use SubprocVecEnv; otherwise DummyVecEnv
    """
    # Load minimal config
    with open("configs/env_configs/small_env_obstacles.yaml", "r") as f:
        env_config = yaml.safe_load(f)
    
    with open("configs/agent_configs/ground_agent.yaml", "r") as f:
        agent_config = yaml.safe_load(f)
    
    with open("configs/target_configs/target_config.yaml", "r") as f:
        target_config = yaml.safe_load(f)
    
    config = {
        **env_config,
        "agents": [agent_config],
        "target_config": target_config,
    }
    
    # Create vectorized environment
    env_fns = [make_env(config) for _ in range(num_envs)]
    
    if use_subproc and num_envs > 1:
        env = SubprocVecEnv(env_fns)
        env_type = "SubprocVecEnv"
    else:
        env = DummyVecEnv(env_fns)
        env_type = "DummyVecEnv"
    
    # Create model
    model = PPO("MlpPolicy", env, verbose=0)
    
    # Benchmark training
    print(f"\nBenchmarking: {num_envs} envs, {env_type}, {timesteps} timesteps")
    start_time = time.time()
    
    model.learn(total_timesteps=timesteps)
    
    elapsed_time = time.time() - start_time
    steps_per_second = timesteps / elapsed_time
    
    print(f"  Time: {elapsed_time:.2f}s")
    print(f"  Speed: {steps_per_second:.2f} steps/sec")
    
    env.close()
    
    return elapsed_time, steps_per_second


def main():
    """Run benchmarks with different configurations."""
    print("=" * 60)
    print("Environment Parallelization Benchmark")
    print("=" * 60)
    
    timesteps = 10000  # Adjust for longer/shorter benchmarks
    configs = [
        (1, False),   # Single env, DummyVecEnv
        (2, True),    # 2 envs, SubprocVecEnv
        (4, True),    # 4 envs, SubprocVecEnv
        (8, True),    # 8 envs, SubprocVecEnv
        (4, False),   # 4 envs, DummyVecEnv (for comparison)
    ]
    
    results = []
    
    for num_envs, use_subproc in configs:
        try:
            elapsed, speed = benchmark(num_envs, timesteps, use_subproc)
            results.append((num_envs, use_subproc, elapsed, speed))
        except Exception as e:
            print(f"  ERROR: {e}")
            results.append((num_envs, use_subproc, None, None))
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Envs':<6} {'Type':<15} {'Time (s)':<12} {'Steps/sec':<12} {'Speedup':<10}")
    print("-" * 60)
    
    baseline_time = None
    for num_envs, use_subproc, elapsed, speed in results:
        env_type = "SubprocVecEnv" if use_subproc else "DummyVecEnv"
        
        if elapsed is None:
            print(f"{num_envs:<6} {env_type:<15} {'FAILED':<12}")
            continue
        
        if baseline_time is None:
            baseline_time = elapsed
            speedup = 1.0
        else:
            speedup = baseline_time / elapsed
        
        print(f"{num_envs:<6} {env_type:<15} {elapsed:<12.2f} {speed:<12.2f} {speedup:<10.2f}x")
    
    print("=" * 60)
    print("\nRecommendation:")
    
    # Find best SubprocVecEnv configuration
    subproc_results = [(n, e, s) for n, use_sub, e, s in results if use_sub and e is not None]
    if subproc_results:
        best = max(subproc_results, key=lambda x: x[2])  # Max speed
        print(f"  Use num_envs = {best[0]} with SubprocVecEnv")
        print(f"  Expected speedup: ~{baseline_time/best[1]:.2f}x faster")
    
    print("\nNote: Actual speedup may vary with longer training runs.")


if __name__ == '__main__':
    main()

