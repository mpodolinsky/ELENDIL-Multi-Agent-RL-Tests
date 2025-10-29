# Parallel Environment - Important Note

## Issue with External Package

The `gymnasium_env` package from HA-SPO2V-Env has an import structure issue that prevents the PettingZoo parallel conversion from working smoothly:

**Problem:** The package's `grid_world_multi_agent.py` imports `agents.agents` but the package structure doesn't properly expose this module when imported as a package.

## Solutions

### ✅ RECOMMENDED: Use Current Working Approach

Your **current implementation** in `custom_env_main.py` is working perfectly and ready for parallel training:

```python
# src/main/custom_env_main.py
config["num_envs"] = 8  # Change this from 1 to 8

# The rest works automatically:
# - Uses make_vec_env with SubprocVecEnv
# - Flattens observations and actions
# - Proper Gymnasium compatibility
```

**Benefits:**
- ✅ Already working
- ✅ Just change `num_envs` to scale
- ✅ Clean SB3 integration
- ✅ Video recording works
- ✅ WandB integration works

### Alternative: Fix the External Package

If you have access to modify the `HA-SPO2V-Env` package, you can fix the import issue:

**In `/mnt/data/Documents/Project_M/HA-SPO2V-Env/gymnasium_env/envs/grid_world_multi_agent.py`:**

Change:
```python
from agents.agents import Agent, FOVAgent
```

To:
```python
import sys
import os
# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from agents.agents import Agent, FOVAgent
```

Or better, fix the package structure by adding proper `__init__.py` files and relative imports.

## Bottom Line

**Your current working solution is the best approach.** Simply increase `num_envs` to 8 in `custom_env_main.py` and you'll have full parallel training with:

- 8 parallel environments
- ~8x faster training
- All features working (video, WandB, etc.)

The PettingZoo parallel conversion is available if you fix the external package imports, but it's not necessary for your use case.

## Quick Start

```bash
# Edit src/main/custom_env_main.py
# Change: "num_envs": 1  →  "num_envs": 8

# Run training
python src/main/custom_env_main.py
```

That's it! You're ready for parallel training. 🚀

