#!/usr/bin/env python3
# %%
# Imports
"""Simple script to run any supervisor on any HuggingFace dataset.

Just modify the variables below and run the script.
"""

import gc
from argparse import ArgumentParser
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

from bells_o import Evaluator, HuggingFaceDataset, Result, Usage
from bells_o.evaluator import DatasetConfig
from bells_o.supervisors import AutoHuggingFaceSupervisor, AutoRestSupervisor


def _coerce_value(value: str) -> str | int | float | bool:
    """Coerce a string value to its most specific scalar type."""
    if value.lower() in ("true", "false"):
        return value.lower() == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _parse_kwargs(items: list[str]) -> dict:
    """Parse a list of 'key=value' strings into a dict with auto-coerced values."""
    result = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Invalid kwarg format: '{item}'. Expected KEY=VALUE.")
        key, value = item.split("=", 1)
        result[key] = _coerce_value(value)
    return result


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        required=False,
        help="If it should evaluate the input or output dataset. Possible values: ['input', 'output'].",
        default="input",
    )
    parser.add_argument("--model-id", type=str, required=False, help="Model identifier for the autoclass (e.g. 'nvidia/llama-3.1-nemotron-safety-guard-8b-v3')")
    parser.add_argument("--type", type=str, required=False, help="rest or hf")
    parser.add_argument("--save_dir", type=str, required=False, help="path to save results in")
    parser.add_argument("--model-kwarg", action="append", metavar="KEY=VALUE", help="Supervisor keyword argument (repeatable, e.g. --model-kwarg backend=vllm)")
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
    supervisor_string = args.model_id or "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"

    supervisor_type = args.type or "hf"
    if supervisor_type == "hf":
        lab, model_name = supervisor_string.split("/")  # for HF supervisors
    else:
        lab = args.lab
        model_name = args.model_name
        if not lab and model_name:
            raise ValueError("For REST supervisors, you need to pass --lab and --model_name.")

    # Parse supervisor kwargs from repeatable --model-kwarg key=value args
    supervisor_kwargs = {"backend": "vllm"} if supervisor_type == "hf" else {}
    supervisor_kwargs |= _parse_kwargs(args.model_kwarg or [])
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

    # Load supervisor
    if supervisor_type == "rest":
        supervisor = AutoRestSupervisor.load(supervisor_string, **supervisor_kwargs)
    elif supervisor_type == "hf":
        supervisor = AutoHuggingFaceSupervisor.load(supervisor_string, **supervisor_kwargs)
    else:
        raise ValueError(f"Unknown supervisor type '{supervisor_type}'. Expected 'rest' or 'hf'.")

    # %%
    # Create evaluator and run
    evaluator = Evaluator(
        dataset_conf,
        supervisor,
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
