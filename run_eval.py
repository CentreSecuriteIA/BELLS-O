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


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="If it should evaluate the input or output dataset. Possible values: ['input', 'output'].",
        default="input",
    )
    parser.add_argument("--name", type=str, required=False, help="name of the model in the autoclass")
    parser.add_argument("--type", type=str, required=False, help="rest or hf")
    parser.add_argument("--save_dir", type=str, required=False, help="path to save results in")
    parser.add_argument("--kwargs", type=str, required=False)
    parser.add_argument("--lab", type=str, required=False, help="The lab name, only for REST supervisors")
    parser.add_argument("--model_name", type=str, required=False, help="The model name, only for REST supervisors")
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size to use, defaults to 1", default=1)
    args = parser.parse_args()

    # %%
    # Script configuration
    # ============================================================================
    # Configuration - modify these variables
    # ============================================================================

    # Dataset configuration

    dataset_name = f"bells-o-project/content-moderation-{args.dataset}"
    usage_type = "content_moderation"
    input_column = "prompt"
    target_column = "category"

    # Supervisor configuration
    supervisor_string = (
        args.name or "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"
    )  # Change this to the according string used in the Auto classes

    supervisor_type = args.type or "hf"
    if supervisor_type == "hf":
        lab, model_name = supervisor_string.split("/")  # for HF supervisors
    else:
        lab = args.lab
        model_name = args.model_name
        if not lab and model_name:
            raise ValueError("For REST supervisors, you need to pass --lab= and --model_name=.")

    # Supervisor kwargs, some need project ids or similar to be specified
    if supervisor_type == "hf":
        supervisor_kwargs = {"backend": "vllm"}
    else:
        supervisor_kwargs = {}
    if args.kwargs is not None:
        supervisor_kwargs |= json.loads(args.kwargs)
    print(f"INFO: Supervisor Kwargs: {supervisor_kwargs}")

    # Output configuration
    save_dir = Path(args.save_dir).resolve() if args.save_dir else Path("results").resolve()
    save_dir_full = save_dir / lab
    run_id = model_name
    verbose = True

    # %%
    # Auxiliary definitions
    # ============================================================================
    # Target mapping function
    # ============================================================================

    def target_map_fn(input: str) -> Result:
        """Map dataset target column to Result."""
        if usage_type == "content_moderation":
            return Result(content_moderation=True if input != "Benign" else False)
        elif usage_type == "jailbreak":
            return Result(jailbreak=True if input.lower() in ["jailbreak", "true", "1"] else False)
        elif usage_type == "prompt_injection":
            return Result(prompt_injection=True if input.lower() in ["injection", "true", "1"] else False)
        else:
            return Result(**{usage_type: True if str(input).lower() in ["true", "1", "yes", "harmful"] else False})

    # ============================================================================
    # Create configurations and run
    # ============================================================================

    # Dataset config
    dataset_conf = DatasetConfig(
        type=HuggingFaceDataset,
        kwargs={
            "name": dataset_name,
            "usage": Usage(usage_type),
            "target_map_fn": target_map_fn,
            "input_column": input_column,
        },
        input_column=input_column,
        target_column=target_column,
    )

    # Supervisor config
    if supervisor_type == "rest":

        class _SupervisorWrapper:  # type: ignore
            def __init__(self, **kwargs):
                self.supervisor = AutoRestSupervisor.load(supervisor_string, **kwargs)

            def __getattr__(self, name):
                return getattr(self.supervisor, name)

            def __call__(self, *args, **kwargs):
                return self.supervisor(*args, **kwargs)

        supervisor_conf = SupervisorConfig(type=_SupervisorWrapper, kwargs=supervisor_kwargs)  # type: ignore
    elif supervisor_type == "hf":

        class _SupervisorWrapper:
            def __init__(self, **kwargs):
                self.supervisor = AutoHuggingFaceSupervisor.load(supervisor_string, **kwargs)

            def __getattr__(self, name):
                return getattr(self.supervisor, name)

            def __call__(self, *args, **kwargs):
                return self.supervisor(*args, **kwargs)

        supervisor_conf = SupervisorConfig(type=_SupervisorWrapper, kwargs=supervisor_kwargs)  # type: ignore
    else:
        raise ValueError("Set either USE_AUTO_REST=True or USE_AUTO_HF=True")
    # %%
    # Create evaluator and run
    evaluator = Evaluator(
        dataset_conf,
        supervisor_conf,
        save_dir=save_dir_full,
        verbose=verbose,
        batch_size=args.batch_size,
    )

    evaluator.run(run_id=run_id, verbose=verbose, save=True)

    print(f"\nDone! Results saved to {save_dir_full}")

    # %%
    del evaluator
    gc.collect()


if __name__ == "__main__":
    main()
