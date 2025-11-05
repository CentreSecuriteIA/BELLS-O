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


type RunDict = dict[str, OutputDict]


def _now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _uuid():
    return str(uuid4())[:8]


# TODO: implement batching, error handling, asynch runs?
@dataclass
class Evaluator:
    """Class that implements structured evaluation of a supervisor on a dataset."""

    def __init__(
        self,
        dataset_config: DatasetConfig,
        supervisor_config: SupervisorConfig,
        metadata: bool = True,  # TODO: customize metadata, e.g. only model data, prompt, date, etc.
        save_dir: str | Path | None = None,
        verbose: bool = False,
    ):
        """Load Evaluator.

        Args:
            dataset_config (DatasetConfig): Config to load dataset.
            supervisor_config (DatasetConfig): Config to load supervisor.
            metadata (bool, optional): If the runs should collect metadata.
            save_dir (str|Path, optional): A path to save the results in. Will create the directory `save_dir` and save results in `save_dir/dataset.name`
            verbose (bool, optional): If a progress bar for runs should be displayed. Defaults to False.

        """
        # set attributes
        self.dataset_config = dataset_config
        self.supervisor_config = supervisor_config
        self.metadata = metadata
        self.verbose = verbose
        self.save_dir: Path | None = (
            save_dir if isinstance(save_dir, Path) or save_dir is None else Path(save_dir)
        )

        # load dataset
        self.dataset = self.dataset_config["type"](**self.dataset_config["kwargs"])
        assert self.dataset.target_map_fn is not None, (  # TODO: make this exhaustive
            "Need `target_map_fn` to be specified for dataset."
        )

        # load supervisor
        self.supervisor = self.supervisor_config["type"](**self.supervisor_config["kwargs"])
        self.runs: dict[str, RunDict] = {}

        self._prepared_dirs = False
        if self.save_dir:
            self._prepare_dirs()
            self._prepared_dirs = True

    # TODO: Doc string
    # TODO: Implement safe runs? (saving a prompt file after every run)
    def run(
        self,
        indices: list[int] | None = None,
        run_id: str | None = None,
        save=False,
        verbose: bool = False,
    ):
        """Run an evaluation on specified indices.

        Args:
            indices (list[int], optional): List of indices of samples in the Dataset to run.
            run_id (str, optional): ID for this run.
            unique (bool, optional): If a sample should be skipped if it was run before.
            save (bool, optional): If the results should be saved after the run. Defaults to False.
            verbose (bool, optional): If a progress bar for the run should be shown.

        """
        verbose = verbose or self.verbose
        if run_id is None:
            run_id = _uuid()
        if run_id in self.runs:
            # log generation new run_id
            while run_id in self.runs:
                run_id = _uuid()
        self.runs[run_id] = {}

        run_dict = self.runs[run_id]

        if indices is None:
            indices = list(range(len(self.dataset)))

        assert indices
        if verbose:
            from tqdm import tqdm

            iterator = tqdm(indices, desc="Processing")
        else:
            iterator = iter(indices)

        started_at = _now()

        for index in iterator:
            sample: dict[str, str] = self.dataset[index]
            prompt_id = sample["prompt_id"]
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
                result_dict["metadata"]["prompt_id"] = prompt_id
                result_dict["metadata"]["prompt"] = prompt
                result_dict["metadata"]["target"] = target
            run_dict[prompt_id] = result_dict

        if self.metadata:
            for output_dict in run_dict.values():
                output_dict["metadata"]["started_at"] = started_at
                output_dict["metadata"]["ended_at"] = _now()
                output_dict["metadata"]["num_prompts"] = len(indices)
                output_dict["metadata"]["supervisor"] = self.supervisor.metadata()
        # TODO: add dataset metadata

        if save:
            self.save_runs()

    # TODO: if implementing safe runs in run(), make this cascaded, so that this calls a function that saves one prompt.
    def save_runs(self, save_dir: str | Path | None = None):
        """Save all current runs to disk.

        Args:
            save_dir (Optional[str|Path]): The path at which to save the runs. Defaults to "runs/".

        """
        save_dir = save_dir or self.save_dir
        if isinstance(save_dir, str):
            save_dir = Path(save_dir)

        if save_dir is None:
            save_dir = Path("runs/")
            print(
                f"WARNING: No valid directory path was provided for saving results. Dumped results to {str(save_dir)}."
            )
        assert isinstance(save_dir, Path)

        if not self._prepare_dirs:
            self._prepare_dirs(save_dir)

        for run_id, run_dict in self.runs.items():
            run_dir = save_dir / self.dataset.clean_name / _clean_string(run_id)
            run_dir.mkdir(exist_ok=True)
            for prompt_id, output_dict in run_dict.items():
                file_path = (run_dir / prompt_id).with_suffix(".json")
                with open(file_path, "w") as f:  # TODO : fix this
                    f.write(json.dumps(output_dict, indent=2))

    def _prepare_dirs(self, save_dir: Path | None = None):
        dataset_name = self.dataset.clean_name

        save_dir = save_dir or self.save_dir
        assert save_dir

        dataset_path = save_dir / dataset_name

        if self.verbose:
            print(f"Create directory {dataset_path} and all necessary parents.")
        dataset_path.mkdir(parents=True, exist_ok=True)


def _clean_string(string: str):
    """Clean a string such that it can be used as a name in a (Windows/UNIX) filesystem."""
    for forbidden_character in '<>:"/\\|?*':
        string = string.replace(forbidden_character, "-")
    return string
