#!/usr/bin/env python3
"""Simple script to run any supervisor on any HuggingFace dataset.
Just modify the variables below and run the script.

To run in the background (continues when screen is locked):
  Option 1: nohup python run_evals.py > output.log 2>&1 &
  Option 2: screen -S evals python run_evals.py  (then detach with Ctrl+A, D)
  Option 3: tmux new -s evals python run_evals.py  (then detach with Ctrl+B, D)

Note: The script will pause if your laptop goes to sleep. To prevent sleep:
  - Keep laptop plugged in and configure System Settings > Energy Saver
  - Or use: caffeinate -i python run_evals.py
"""

from bells_o import Evaluator, HuggingFaceDataset, Result, Usage
from bells_o.evaluator import DatasetConfig, SupervisorConfig
from bells_o.supervisors import AutoHuggingFaceSupervisor, AutoRestSupervisor
from dotenv import load_dotenv

load_dotenv()

# ============================================================================
# Configuration - modify these variables
# ============================================================================

# Dataset configuration
DATASET_NAME = "bellsop/BELLS-O_Dataset"
USAGE_TYPE = "content_moderation"  # or "jailbreak", "prompt_injection"
INPUT_COLUMN = "prompt"
TARGET_COLUMN = "category"

# Supervisor configuration
# Option 1: Use AutoRestSupervisor (for REST API supervisors)
# Examples: "google-moderation", "anthropic-classification", "mistral-classification"
SUPERVISOR_TYPE = "aegis-defensive-1.0"  # Change this
USE_AUTO_REST = False
USE_AUTO_HF = True

# Option 2: Use AutoHuggingFaceSupervisor (for HF model supervisors)
# Examples: "meta-llama/Llama-Guard-4-12B", "openai/gpt-oss-safeguard-20b"
# SUPERVISOR_TYPE = "meta-llama/Llama-Gua   rd-4-12B"
# USE_AUTO_REST = False
# USE_AUTO_HF = True

# Supervisor kwargs (e.g., {"api_key": "..."} or {"api_variable": "GEMINI_API_KEY"})
SUPERVISOR_KWARGS = {}

# Output configuration
SAVE_DIR = "results/"
SAVE_DIR_FULL = SAVE_DIR + "nvidia_moderation"
RUN_ID = "shieldgemma_9b"
VERBOSE = True

# ============================================================================
# Target mapping function
# ============================================================================

def target_map_fn(input: str) -> Result:
    """Map dataset target column to Result."""
    if USAGE_TYPE == "content_moderation":
        return Result(content_moderation=True if input != "Benign" else False)
    elif USAGE_TYPE == "jailbreak":
        return Result(jailbreak=True if input.lower() in ["jailbreak", "true", "1"] else False)
    elif USAGE_TYPE == "prompt_injection":
        return Result(prompt_injection=True if input.lower() in ["injection", "true", "1"] else False)
    else:
        return Result(**{USAGE_TYPE: True if str(input).lower() in ["true", "1", "yes", "harmful"] else False})


# ============================================================================
# Create configurations and run
# ============================================================================

# Dataset config
dataset_conf = DatasetConfig(
    type=HuggingFaceDataset,
    kwargs={
        "name": DATASET_NAME,
        "usage": Usage(USAGE_TYPE),
        "target_map_fn": target_map_fn,
    },
    input_column=INPUT_COLUMN,
    target_column=TARGET_COLUMN,
)

# Supervisor config
if USE_AUTO_REST:
    class SupervisorWrapper:
        def __init__(self, **kwargs):
            self.supervisor = AutoRestSupervisor.load(SUPERVISOR_TYPE, **kwargs)
        def __getattr__(self, name):
            return getattr(self.supervisor, name)
        def __call__(self, *args, **kwargs):
            return self.supervisor(*args, **kwargs)
    supervisor_conf = SupervisorConfig(type=SupervisorWrapper, kwargs=SUPERVISOR_KWARGS)
elif USE_AUTO_HF:
    class SupervisorWrapper:
        def __init__(self, **kwargs):
            self.supervisor = AutoHuggingFaceSupervisor.load(SUPERVISOR_TYPE, **kwargs)
        def __getattr__(self, name):
            return getattr(self.supervisor, name)
        def __call__(self, *args, **kwargs):
            return self.supervisor(*args, **kwargs)
    supervisor_conf = SupervisorConfig(type=SupervisorWrapper, kwargs=SUPERVISOR_KWARGS)
else:
    raise ValueError("Set either USE_AUTO_REST=True or USE_AUTO_HF=True")

# Create evaluator and run
evaluator = Evaluator(
    dataset_conf,
    supervisor_conf,
    save_dir=SAVE_DIR_FULL,
    verbose=VERBOSE,
)

evaluator.run(run_id=RUN_ID, verbose=VERBOSE, save=True)

print(f"\nDone! Results saved to {SAVE_DIR_FULL}")
