"""Utility functions for model evaluation."""

import os
import glob
import subprocess
import sys


def evaluate_checkpointed_models(run_id, num_episodes=5):
    """Find all checkpointed models with _0_ pattern and evaluate them.
    
    Args:
        run_id: The WandB run ID to evaluate checkpoints for
        num_episodes: Number of episodes to evaluate each checkpoint
    """
    import re
    
    # Find all checkpoint files with _0_ pattern in the current run directory
    # Check both wandb/latest-run/ and wandb/run-*-{run_id}/ directories
    checkpoint_patterns = [
        f"wandb/latest-run/_0_*.pt",
        f"wandb/run-*-{run_id}/_0_*.pt"
    ]
    
    checkpoint_files = []
    for pattern in checkpoint_patterns:
        checkpoint_files.extend(glob.glob(pattern))
    
    # Remove duplicates while preserving order
    seen = set()
    checkpoint_files = [f for f in checkpoint_files if f not in seen and not seen.add(f)]
    
    if not checkpoint_files:
        print(f"No checkpoint files found matching patterns")
        return
    
    # Sort by step number (extract from filename)
    def extract_step(filename):
        match = re.search(r'_0_(\d+)\.pt$', filename)
        return int(match.group(1)) if match else 0
    
    checkpoint_files.sort(key=extract_step)
    
    print(f"\n{'='*70}")
    print(f"Found {len(checkpoint_files)} checkpoint files to evaluate")
    print(f"{'='*70}\n")
    
    # Path to the evaluation script
    eval_script_path = "src/elendil/test/agilerl_maddpg_eval.py"
    
    # Evaluate each checkpoint
    for checkpoint_file in checkpoint_files:
        # Extract model ID from filename (e.g., _0_100000 -> _0_100000)
        model_id = os.path.basename(checkpoint_file).replace('.pt', '')
        
        print(f"\n{'='*70}")
        print(f"Evaluating checkpoint: {model_id}")
        print(f"File: {checkpoint_file}")
        print(f"{'='*70}")
        
        try:
            # Run evaluation script
            result = subprocess.run(
                [sys.executable, eval_script_path, model_id, "--num_episodes", str(num_episodes)],
                cwd=os.getcwd(),
                capture_output=False,
                text=True
            )
            
            if result.returncode == 0:
                print(f"✅ Successfully evaluated {model_id}")
            else:
                print(f"⚠️  Evaluation of {model_id} completed with return code {result.returncode}")
                
        except Exception as e:
            print(f"❌ Error evaluating {model_id}: {e}")
            continue
    
    print(f"\n{'='*70}")
    print(f"Finished evaluating all checkpoints")
    print(f"{'='*70}\n")

