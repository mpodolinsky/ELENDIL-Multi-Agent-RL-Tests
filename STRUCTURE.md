# Repository Structure

This repository contains two main components for heterogeneous multi-agent reinforcement learning:

## 📁 Directory Structure

```
heterogeneous-marl/
├── hemac/                          # HeMAC (Heterogeneous Multi-Agent Control)
│   ├── train/                      # HeMAC training scripts
│   │   └── simple_push_v3_train.py # IPPO training with AgileRL
│   ├── test/                       # HeMAC testing scripts
│   │   └── simple_push_v3_test.py  # Model evaluation scripts
│   └── docs/                       # HeMAC documentation
├── src/                            # ELENDIL environment
│   ├── elendil/                    # ELENDIL package
│   │   ├── main/                   # Main training scripts
│   │   │   ├── custom_env_main.py  # Main ELENDIL training script
│   │   │   ├── benchmark_parallel.py
│   │   │   ├── parallel_train.py
│   │   │   ├── test_parallel_env.py
│   │   │   └── eval.py
│   │   ├── envs/                   # Environment implementations
│   │   │   ├── gridworld.py
│   │   │   └── parallel_gridworld.py
│   │   ├── wrappers/               # Environment wrappers
│   │   │   ├── observation_wrapper.py
│   │   │   └── aec_parallel_wrapper.py
│   │   └── docs/                   # ELENDIL documentation
│   │       ├── PARALLELIZATION.md
│   │       ├── MULTIPROCESSING_FIX.md
│   │       ├── MAKE_VEC_ENV_MIGRATION.md
│   │       ├── PARALLEL_ENV_GUIDE.md
│   │       ├── PARALLEL_ENV_NOTE.md
│   ├── agents/                     # Agent implementations
│   ├── train/                      # General training utilities
│   ├── test/                       # General testing utilities
│   └── utils/                      # Utility functions
├── configs/                        # Configuration files
│   ├── agent_configs/
│   ├── env_configs/
│   └── target_configs/
├── data/                           # Data storage
├── models/                         # Model checkpoints
├── results/                        # Experiment results
├── runs/                           # TensorBoard runs
├── videos/                         # Training videos
└── wandb/                          # WandB logs
```

## 🎯 Components

### HeMAC (Heterogeneous Multi-Agent Control)
- **Location**: `hemac/`
- **Purpose**: HeMAC-specific training and testing using AgileRL and IPPO
- **Environments**: Multi-Particle Environment (MPE) variants
- **Algorithms**: IPPO (Independent Proximal Policy Optimization)

### ELENDIL Environment
- **Location**: `src/elendil/`
- **Purpose**: Custom gridworld environment for heterogeneous agents
- **Features**: 
  - PettingZoo AEC/Parallel environment support
  - Custom observation/action wrappers
  - Multi-agent training with Stable-Baselines3
- **Algorithms**: PPO, IPPO, MAPPO (planned)

## 🚀 Quick Start

### HeMAC Training
```bash
cd hemac/train
python simple_push_v3_train.py
```

### ELENDIL Training
```bash
cd src/elendil/main
python custom_env_main.py
```

## 📚 Documentation
- HeMAC docs: `hemac/docs/`
- ELENDIL docs: `src/elendil/docs/`
- Main README: `README.md`
