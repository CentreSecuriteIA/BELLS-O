#!/usr/bin/env python3
# %%
# Imports
"""Simple script to run any supervisor on any HuggingFace dataset.

Just modify the variables below and run the script.
"""

import gc
import json
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

from bells_o import Evaluator, HuggingFaceDataset, Result, Usage
from bells_o.evaluator import DatasetConfig, SupervisorConfig
from bells_o.supervisors import AutoHuggingFaceSupervisor, AutoRestSupervisor


DATASET_NAME = "bells-o-project/content-moderation-input"
USAGE_TYPE = "content_moderation"  # or "jailbreak", "prompt_injection"
INPUT_COLUMN = "prompt"
TARGET_COLUMN = "category"


def main():
    parser = ArgumentParser()
    parser.add_argument("--name", type=str, required=False, help="name of the model in the autoclass")
    parser.add_argument("--type", type=str, required=False, help="rest or hf")
    parser.add_argument("--save_dir", type=str, required=False, help="path to save results in")
    parser.add_argument("--kwargs", type=str, required=False)
    args = parser.parse_args()

    # %%
    # Script configuration
    # ============================================================================
    # Configuration - modify these variables
    # ============================================================================

    # Dataset configuration

    # Supervisor configuration
    SUPERVISOR_STRING = (
        args.name or "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
    )  # Change this to the according string used in the Auto classes
    lab, model_name = SUPERVISOR_STRING.split("/")  # for HF supervisors

    SUPERVISOR_TYPE = args.type or "hf"

    # Supervisor kwargs, some need project ids or similar to be specified
    SUPERVISOR_KWARGS = {"backend": "vllm"}
    if args.kwargs is not None:
        SUPERVISOR_KWARGS |= json.loads(args.kwargs)

    # Output configuration
    SAVE_DIR = Path(args.save_dir).resolve() if args.save_dir else Path("results").resolve()
    SAVE_DIR_FULL = SAVE_DIR / lab
    RUN_ID = model_name
    VERBOSE = True

    # %%
    # Auxiliary definitions
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
            "input_column": INPUT_COLUMN,
        },
        input_column=INPUT_COLUMN,
        target_column=TARGET_COLUMN,
    )

    # Supervisor config
    if SUPERVISOR_TYPE == "rest":

        class _SupervisorWrapper:  # type: ignore
            def __init__(self, **kwargs):
                self.supervisor = AutoRestSupervisor.load(SUPERVISOR_STRING, **kwargs)

            def __getattr__(self, name):
                return getattr(self.supervisor, name)

            def __call__(self, *args, **kwargs):
                return self.supervisor(*args, **kwargs)

        supervisor_conf = SupervisorConfig(type=_SupervisorWrapper, kwargs=SUPERVISOR_KWARGS)  # type: ignore
    elif SUPERVISOR_TYPE == "hf":

        class _SupervisorWrapper:
            def __init__(self, **kwargs):
                self.supervisor = AutoHuggingFaceSupervisor.load(SUPERVISOR_STRING, **kwargs)

            def __getattr__(self, name):
                return getattr(self.supervisor, name)

            def __call__(self, *args, **kwargs):
                return self.supervisor(*args, **kwargs)

        supervisor_conf = SupervisorConfig(type=_SupervisorWrapper, kwargs=SUPERVISOR_KWARGS)  # type: ignore
    else:
        raise ValueError("Set either USE_AUTO_REST=True or USE_AUTO_HF=True")
    # %%
    # Create evaluator and run
    evaluator = Evaluator(
        dataset_conf,
        supervisor_conf,
        save_dir=SAVE_DIR_FULL,
        verbose=VERBOSE,
    )

    evaluator.run(run_id=RUN_ID, verbose=VERBOSE, save=True)

    print(f"\nDone! Results saved to {SAVE_DIR_FULL}")

    # %%
    del evaluator
    gc.collect()


if __name__ == "__main__":
    main()
