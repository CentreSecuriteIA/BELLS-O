"""Implement a structured evaluation class."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Type, TypedDict
from uuid import uuid4

from bells_o.common import OutputDict
from bells_o.datasets import Dataset
from bells_o.supervisors import Supervisor


class DatasetConfig(TypedDict):
    type: Type[Dataset]
    kwargs: dict[str, Any]
    input_column: str
    target_column: str


class SupervisorConfig(TypedDict):
    type: Type[Supervisor]
    kwargs: dict[str, Any]


class RunDict(TypedDict):
    results: list[OutputDict]
    metadata: dict[str, Any]


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# TODO: implement batching, error handling
@dataclass
class Evaluator:
    dataset_config: DatasetConfig
    supervisor_config: SupervisorConfig
    metadata: bool = True  # TODO: customize metadata, e.g. only model data, prompt, date, etc.
    verbose: bool = False
    save_dir: str | Path | None = None

    def __post_init__(self):
        """Load the dataset and supervisor."""
        self.dataset = self.dataset_config["type"](**self.dataset_config["kwargs"])
        assert self.dataset.target_map_fn is not None, (
            "Need `targer_map_fn` to be specified for dataset."
        )
        self.supervisor = self.supervisor_config["type"](**self.supervisor_config["kwargs"])
        self.runs: dict[str, RunDict] = {}
        if isinstance(self.save_dir, str):
            self.save_dir = Path(self.save_dir)

    def run(self, indices: list[int] | None = None, run_id: str | None = None, verbose=False):
        """Run an evaluation on specified indices."""
        verbose = verbose or self.verbose
        if run_id is None:
            run_id = str(uuid4())[:8]
        if run_id in self.runs:
            # log generation new run_id
            while run_id in self.runs:
                run_id = str(uuid4())[:8]
        self.runs[run_id] = RunDict(results=[], metadata={"run_id": run_id, "started_at": _now()})

        if indices is None:
            indices = list(range(len(self.dataset)))

        assert indices
        if verbose:
            from tqdm import tqdm

            iterator = tqdm(indices, desc="Processing")
        else:
            iterator = iter(indices)

        for index in iterator:
            sample: dict[str, str] = self.dataset[index]
            prompt = sample[self.dataset_config["input_column"]]
            target = sample[self.dataset_config["target_column"]]

            # run inference
            result_dict = self.supervisor(prompt)[0]

            assert self.dataset.target_map_fn is not None, (
                "Need `targer_map_fn` to be specified for dataset."
            )
            result_dict["target_result"] = self.dataset.target_map_fn(target)

            # check output against target
            assert "output_result" in result_dict
            result_dict["is_correct"] = result_dict["output_result"] == result_dict["target_result"]

            # add metadata if requested
            if self.metadata:
                result_dict["metadata"]["date"] = _now()
                result_dict["metadata"]["prompt"] = prompt
                result_dict["metadata"]["target"] = target
            self.runs[run_id]["results"].append(result_dict)

        if self.metadata:
            self.runs[run_id]["metadata"]["ended_at"] = _now()
            self.runs[run_id]["metadata"]["num_prompts"] = len(indices)
            self.runs[run_id]["metadata"]["supervisor"] = self.supervisor.metadata()
        # TODO: add dataset metadata
        # TODO:

    def save_runs(self, save_dir: str | Path | None = None):
        """Save all current runs to disk.

        Args:
            save_dir (Optional[str|Path]): The path at which to save the runs. Defaults to "runs/".

        """
        if save_dir is None:
            save_dir = self.save_dir or Path("runs/")
        assert isinstance(save_dir, Path)
        save_dir.mkdir(parents=True, exist_ok=True)
        for run_id, run in self.runs.items():
            with open((save_dir / run_id).with_suffix(".json"), "w") as f:  # TODO : fix this
                f.write(json.dumps(run, indent=2))
