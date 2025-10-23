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
   python src/main/custom_env_main.py
   ```

## Structure

- `src/main/` - Main training scripts
- `src/envs/` - Environment definitions
- `src/agents/` - Agent implementations
- `src/train/` - Training utilities
- `src/test/` - Evaluation scripts
- `configs/` - Configuration files