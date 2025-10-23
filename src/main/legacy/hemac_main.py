# IMPORTS
import yaml
import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.test.hemac_test import record_tests
from train.legacy.hemac_train import train
import torch

# Completely disable X11
os.environ["SDL_VIDEODRIVER"] = "dummy"
os.environ.pop("DISPLAY", None)  # remove any existing X11 display variable
os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"

# SETUP FROM CONFIG FILE

ippo_config_path = "ippo_default_config_HeMAC.yaml"
hemac_config_path = "hemac_env_config.yaml"

# TRAINING

train(ippo_config_path, hemac_config_path, project_name="HeMAC-1q1o-1sP-std-r", run_name="ippo_discrete_actions_batch_4096_900_cycles_large_max_grad_1_learnstep_16384")

# EVALUATION

record_tests(hemac_config=hemac_config_path, NUM_EPISODES=5, VIDEO_FOLDER="wandb/latest-run/videos")

# Final message
print("Evaluation completed. Videos saved to wandb/latest-run/videos.")

