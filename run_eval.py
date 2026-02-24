#!/usr/bin/env python3
# Imports
"""Simple script to run any supervisor on any HuggingFace dataset.

Datasets can be configured either via a JSON config file (--config) or
inline flags (--dataset-id, --usage, --input-column, --target-column).
"""

import gc
import json
from argparse import ArgumentParser
from pathlib import Path
from typing import Any

from dotenv import load_dotenv


load_dotenv()

from bells_o import Evaluator, HuggingFaceDataset, Result, Usage
from bells_o.evaluator import DatasetConfig
from bells_o.supervisors import AutoCustomSupervisor, AutoHuggingFaceSupervisor, AutoRestSupervisor


TRUTHY_DEFAULTS: dict[str, list[str]] = {
    "content_moderation": ["!Benign"],
    "jailbreak": ["jailbreak", "true", "1"],
    "prompt_injection": ["injection", "true", "1"],
}


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


def _make_target_map_fn(usage_type: str, truthy_values: list[str] | None = None):
    """Build a target_map_fn from a usage type and optional truthy value list.

    Values prefixed with '!' are negated (everything except that value is truthy).
    Without prefix, the listed values are the truthy ones. Falls back to
    TRUTHY_DEFAULTS for known usage types, or generic truthiness otherwise.
    """
    truthy = truthy_values or TRUTHY_DEFAULTS.get(usage_type, ["true", "1", "yes", "harmful"])

    negated = [v[1:] for v in truthy if v.startswith("!")]
    positive = [v.lower() for v in truthy if not v.startswith("!")]

    def target_map_fn(input: str) -> Result:
        if negated:
            is_truthy = input not in negated
        else:
            is_truthy = str(input).lower() in positive
        return Result(**{usage_type: is_truthy})

    return target_map_fn


def _build_dataset_config(entry: dict) -> DatasetConfig:
    """Build a DatasetConfig from a dict (config file entry or parsed CLI args)."""
    usage_type = entry["usage"]
    input_column = entry.get("input_column", "prompt")
    target_column = entry.get("target_column", "category")
    target_map_fn = _make_target_map_fn(usage_type, entry.get("truthy_values"))

    kwargs = {
        "name": entry["dataset_id"],
        "usage": Usage(usage_type),
        "target_map_fn": target_map_fn,
        "input_column": input_column,
    }
    if "version_name" in entry:
        kwargs["version_name"] = entry["version_name"]

    return DatasetConfig(
        type=HuggingFaceDataset,
        kwargs=kwargs,
        input_column=input_column,
        target_column=target_column,
    )


def main():
    parser = ArgumentParser()

    # Dataset configuration (mutually exclusive: --config or inline flags)
    dataset_group = parser.add_argument_group("dataset configuration")
    dataset_group.add_argument(
        "--config", type=str, required=False, help="Path to a JSON config file defining one or more datasets"
    )
    dataset_group.add_argument(
        "--dataset-id",
        type=str,
        required=False,
        help="HuggingFace dataset identifier (e.g. 'bells-o-project/content-moderation-input')",
    )
    dataset_group.add_argument(
        "--usage", type=str, required=False, help="Usage type (e.g. 'content_moderation', 'jailbreak')"
    )
    dataset_group.add_argument(
        "--input-column",
        type=str,
        required=False,
        default="prompt",
        help="Name of the input column in the dataset (default: 'prompt')",
    )
    dataset_group.add_argument(
        "--target-column",
        type=str,
        required=False,
        default="category",
        help="Name of the target column in the dataset (default: 'category')",
    )
    dataset_group.add_argument(
        "--truthy-value",
        action="append",
        metavar="VALUE",
        help="Values in the target column that map to True (repeatable, prefix with '!' to negate, e.g. --truthy-value '!Benign')",
    )

    # Supervisor configuration
    parser.add_argument(
        "--model-id",
        type=str,
        required=False,
        help="Model identifier for the autoclass (e.g. 'nvidia/llama-3.1-nemotron-safety-guard-8b-v3')",
    )
    parser.add_argument("--type", type=str, required=False, help="rest or hf")
    parser.add_argument(
        "--supervisor-kwarg",
        action="append",
        metavar="KEY=VALUE",
        help="Supervisor keyword argument (repeatable, e.g. --supervisor-kwarg backend=vllm)",
    )
    parser.add_argument("--lab", type=str, required=False, help="The lab name, only for REST supervisors")
    parser.add_argument("--model_name", type=str, required=False, help="The model name, only for REST supervisors")

    # Run configuration
    parser.add_argument("--save_dir", type=str, required=False, help="path to save results in")
    parser.add_argument("--batch_size", type=int, required=False, help="Batch size to use, defaults to 1", default=1)
    args = parser.parse_args()

    # Dataset configuration
    if args.config:
        with open(args.config) as f:
            config_data = json.load(f)
        if isinstance(config_data, dict):
            config_data = [config_data]
        dataset_configs = [_build_dataset_config(entry) for entry in config_data]
    elif args.dataset_id and args.usage:
        entry = {
            "dataset_id": args.dataset_id,
            "usage": args.usage,
            "input_column": args.input_column,
            "target_column": args.target_column,
        }
        if args.truthy_value:
            entry["truthy_values"] = args.truthy_value
        dataset_configs = [_build_dataset_config(entry)]
    else:
        parser.error("Provide either --config or both --dataset-id and --usage.")

    # Supervisor configuration
    supervisor_type = args.type or "hf"

    # Parse supervisor kwargs from repeatable --model-kwarg key=value args
    supervisor_kwargs: dict[str, Any] = {"backend": "vllm"} if supervisor_type == "hf" else {}
    supervisor_kwargs |= _parse_kwargs(args.supervisor_kwarg or [])

    if supervisor_kwargs["backend"] == "transformers":
        supervisor_kwargs["model_kwargs"] = {"device_map": "auto"}

    supervisor_string = args.model_id or "nvidia/llama-3.1-nemotron-safety-guard-8b-v3"

    print(f"INFO: Model ID: {supervisor_string}, Supervisor Kwargs: {supervisor_kwargs}")

    # Load supervisor
    if supervisor_type == "hf":
        lab, model_name = supervisor_string.split("/")  # for HF supervisors
        supervisor = AutoHuggingFaceSupervisor.load(supervisor_string, **supervisor_kwargs)
        print(f"DEBUG: Supervisor device: {supervisor._model.device}")
    elif supervisor_type == "rest":
        lab = args.lab
        model_name = args.model_name
        if not lab and model_name:
            raise ValueError("For REST supervisors, you need to pass --lab and --model_name.")
        supervisor = AutoRestSupervisor.load(supervisor_string, **supervisor_kwargs)
    elif supervisor_type == "custom":
        lab, model_name = supervisor_string.split("/")  # for HF supervisors
        supervisor = AutoCustomSupervisor.load(supervisor_string, **supervisor_kwargs)
    else:
        raise ValueError(f"Unknown supervisor type '{supervisor_type}'. Expected 'hf', 'rest', or 'custom'.")

    # Output configuration
    save_dir = Path(args.save_dir).resolve() if args.save_dir else Path("results").resolve()
    save_dir_full = save_dir / lab
    run_id = model_name
    verbose = True

    # Create evaluator and run
    evaluator = Evaluator(
        dataset_configs,
        supervisor,
        save_dir=save_dir_full,
        verbose=verbose,
        batch_size=args.batch_size,
    )

    evaluator.run(run_id=run_id, verbose=verbose, save=True)

    print(f"\nDone! Results saved to {save_dir_full}")

    del evaluator
    gc.collect()


if __name__ == "__main__":
    main()
