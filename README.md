# heterogeneous-marl

Research, training, and testing repository for multi-agent reinforcement learning (MARL) on the **HA-SPO2V** (Heterogeneous Agent - Stealth Partially Observable Variable Viewpoint) environment.

## Overview

This repository implements and evaluates various MARL algorithms for heterogeneous multi-agent teaming scenarios, focusing on improving stability, reliability, and speed of cooperative behaviors.

## Environment

**HA-SPO2V**: A gridworld environment featuring heterogeneous agents with asymmetric capabilities and partial observability working together to achieve common goals.

## Quick Start

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Activate virtual environment:**
   ```bash
   source .venv/bin/activate
   ```

3. **Train agents:**
   ```bash
   # ELENDIL environment training
   python src/elendil/main/custom_env_main.py
   
   # HeMAC training (if using AgileRL)
   python hemac/train/simple_push_v3_train.py
   ```

## Structure

### HeMAC (Heterogeneous Multi-Agent Control)
- `hemac/train/` - HeMAC training scripts (IPPO with AgileRL)
- `hemac/test/` - HeMAC evaluation scripts
- `hemac/docs/` - HeMAC documentation

### ELENDIL Environment
- `src/elendil/main/` - Main training scripts
- `src/elendil/envs/` - Environment definitions
- `src/elendil/wrappers/` - Environment wrappers
- `src/elendil/docs/` - ELENDIL documentation

### General
- `src/agents/` - Agent implementations
- `src/train/` - Training utilities
- `src/test/` - Evaluation scripts
- `configs/` - Configuration files

See `STRUCTURE.md` for detailed directory organization.