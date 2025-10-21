"""Implement a structured evaluation class."""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Type, TypedDict
from uuid import uuid4

from bells_o import Dataset, OutputDict, Supervisor


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
    target_map_fn: Callable
    metadata: bool = True  # TODO: customize metadata, e.g. only model data, prompt, date, etc.
    save_dir: str | Path | None = None

    def __post_init__(self):
        """Load the dataset and supervisor."""
        self.dataset = self.dataset_config["type"](**self.dataset_config["kwargs"])
        self.supervisor = self.supervisor_config["type"](**self.supervisor_config["kwargs"])
        self.runs: dict[str, RunDict] = {}
        if isinstance(self.save_dir, str):
            self.save_dir = Path(self.save_dir)

    def run(self, indices: list[int] | None = None, run_id: str | None = None):
        """Run an evaluation on specified indices."""
        if run_id is None:
            run_id = str(uuid4())
        if run_id in self.runs:
            # log generation new run_id
            while run_id in self.runs:
                run_id = str(uuid4())
        self.runs[run_id] = RunDict(results=[], metadata={"started_at": _now()})

        if indices is None:
            indices = list(range(len(self.dataset)))
        for index in indices:
            sample: dict[str, str] = self.dataset[index]
            prompt = sample[self.dataset_config["input_column"]]
            target = sample[self.dataset_config["target_column"]]

            # run inference
            result_dict = self.supervisor(prompt)[0]
            result_dict["target_result"] = self.target_map_fn(target)

            # check output against target
            assert "output_result" in result_dict
            result_dict["is_correct"] = result_dict["output_result"] == result_dict["target_result"]

            # add metadata if requested
            if self.metadata:
                result_dict["metadata"]["prompt"] = prompt
                result_dict["metadata"]["date"] = _now()
            self.runs[run_id]["results"].append(result_dict)

        if self.metadata:
            self.runs[run_id]["metadata"]["ended_at"] = _now()
            self.runs[run_id]["metadata"]["num_prompts"] = len(indices)
            self.runs[run_id]["metadata"]["supervisor"] = self.supervisor.metadata()
        # TODO: add dataset metadata

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
            with open(save_dir / run_id / ".json", "w") as f:
                f.write(json.dumps(run))
