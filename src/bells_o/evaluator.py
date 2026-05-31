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


# TODO: implement the Auto classes and make the interface nicer
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


# TODO: implement error handling, asynch runs?
@dataclass
class Evaluator:
    """Class that implements structured evaluation of a supervisor on one or more datasets."""

    # TODO: preemptive checking if there are unrun samples. if not, do not load the supervisor.
    def __init__(
        self,
        dataset_configs: DatasetConfig | list[DatasetConfig],
        supervisor: SupervisorConfig | Supervisor,
        metadata: bool = True,  # TODO: customize metadata, e.g. only model data, prompt, date, etc.
        save_dir: str | Path | None = None,
        verbose: bool = False,
        batch_size: int = 1,
    ):
        """Load Evaluator.

        Args:
            dataset_configs (DatasetConfig | list[DatasetConfig]): Config(s) to load dataset(s).
            supervisor (SupervisorConfig | Supervisor): A Supervisor instance or a config to construct one.
            metadata (bool, optional): If the runs should collect metadata.
            save_dir (str|Path, optional): A path to save the results in. Results are saved under `save_dir/dataset.clean_name/`.
            verbose (bool, optional): If a progress bar for runs should be displayed. Defaults to False.
            batch_size (int, optional): Number of prompts to process per batch. Defaults to 1.

        """
        # set attributes
        self._batch_size = batch_size
        if isinstance(dataset_configs, list):
            self.dataset_configs = dataset_configs
        else:
            self.dataset_configs = [dataset_configs]
        self.metadata = metadata
        self.verbose = verbose
        self.save_dir: Path | None = save_dir if isinstance(save_dir, Path) or save_dir is None else Path(save_dir)

        # load datasets
        self.datasets: list[Dataset] = []
        for config in self.dataset_configs:
            dataset = config["type"](**config["kwargs"])
            assert dataset.target_map_fn is not None, (  # TODO: make this exhaustive
                "Need `target_map_fn` to be specified for dataset."
            )
            self.datasets.append(dataset)

        # load or assign supervisor
        if isinstance(supervisor, Supervisor):
            self.supervisor = supervisor
        else:
            self.supervisor = supervisor["type"](**supervisor["kwargs"])

        # runs[dataset_name][run_id] = {prompt_id: output_dict, ...}
        self.runs: dict[str, dict[str, RunDict]] = {}

        self._prepared_dirs = False
        if self.save_dir:
            self._prepare_dirs()
            self._prepared_dirs = True

    def run(
        self,
        indices: list[int] | None = None,
        run_id: str | None = None,
        save=False,
        verbose: bool = False,
    ):
        """Run an evaluation on all datasets.

        Args:
            indices (list[int], optional): List of indices of samples to run. Applied per dataset. If None, all samples are used.
            run_id (str, optional): ID for this run.
            save (bool, optional): If the results should be saved after the run. Defaults to False.
            verbose (bool, optional): If a progress bar for the run should be shown.

        """
        for dataset, dataset_config in zip(self.datasets, self.dataset_configs):
            self._run_dataset(dataset, dataset_config, indices, run_id, save, verbose)

    def _run_dataset(
        self,
        dataset: Dataset,
        dataset_config: DatasetConfig,
        indices: list[int] | None = None,
        run_id: str | None = None,
        save=False,
        verbose: bool = False,
    ):
        """Run an evaluation on a single dataset.

        Args:
            dataset (Dataset): The dataset to evaluate on.
            dataset_config (DatasetConfig): The config for this dataset.
            indices (list[int], optional): List of indices of samples in the Dataset to run.
            run_id (str, optional): ID for this run.
            save (bool, optional): If the results should be saved after the run. Defaults to False.
            verbose (bool, optional): If a progress bar for the run should be shown.

        """
        verbose = verbose or self.verbose
        dataset_name = dataset.clean_name

        # Ensure this dataset has an entry in runs
        if dataset_name not in self.runs:
            self.runs[dataset_name] = {}

        if run_id is None:
            run_id = _uuid()
        if run_id in self.runs[dataset_name]:
            while run_id in self.runs[dataset_name]:
                run_id = _uuid()
        self.runs[dataset_name][run_id] = {}

        run_dict = self.runs[dataset_name][run_id]

        if indices is None:
            indices = list(range(len(dataset)))

        assert indices

        # Ensure save_dir is set up if we're saving
        if save and self.save_dir:
            if not self._prepared_dirs:
                self._prepare_dirs()
                self._prepared_dirs = True

        if verbose:
            from tqdm import tqdm

            iterator = tqdm(indices, desc=f"Processing {dataset_name}")
        else:
            iterator = iter(indices)

        started_at = _now()
        processed_count = 0
        skipped_count = 0

        # Process in batches
        batch = []

        for index in iterator:
            sample: dict[str, str] = dataset[index]
            prompt_id = sample["prompt_id"]
            prompt = sample[dataset_config["input_column"]]
            target = sample[dataset_config["target_column"]]

            # Check if result already exists
            existing_result = self._load_existing_result(dataset_name, prompt_id, run_id)
            if existing_result is not None:
                run_dict[prompt_id] = existing_result
                skipped_count += 1
                if verbose:
                    iterator.set_postfix({"skipped": skipped_count, "processed": processed_count})
                continue

            # Add to batch
            batch.append({"prompt": prompt, "prompt_id": prompt_id, "target": target})

            # Process batch when full
            if len(batch) >= self._batch_size:
                processed_count += self._process_batch(batch, dataset, dataset_name, run_dict, run_id, save)
                if verbose:
                    iterator.set_postfix({"skipped": skipped_count, "processed": processed_count})
                batch = []

        # Process remaining items in final batch
        if batch:
            processed_count += self._process_batch(batch, dataset, dataset_name, run_dict, run_id, save)
            if verbose:
                iterator.set_postfix({"skipped": skipped_count, "processed": processed_count})

        # Update metadata for all results in the run
        if self.metadata:
            for output_dict in run_dict.values():
                output_dict["metadata"]["started_at"] = started_at
                output_dict["metadata"]["ended_at"] = _now()
                output_dict["metadata"]["num_prompts"] = len(indices)
                # Re-save if we're saving iteratively to update metadata
                if save and "prompt_id" in output_dict.get("metadata", {}):
                    self._save_single_result(dataset_name, output_dict["metadata"]["prompt_id"], run_id, output_dict)

    def _process_batch(
        self,
        batch: list[dict],
        dataset: Dataset,
        dataset_name: str,
        run_dict: RunDict,
        run_id: str,
        save: bool,
    ) -> int:
        """Process a batch of prompts through the supervisor and record results.

        Returns:
            Number of prompts processed.

        """
        prompts = [item["prompt"] for item in batch]
        result_dicts = self.supervisor(prompts)
        processed = 0

        for item, result_dict in zip(batch, result_dicts):
            assert dataset.target_map_fn is not None, "Need `target_map_fn` to be specified for dataset."
            result_dict["target_result"] = dataset.target_map_fn(item["target"])

            assert "output_result" in result_dict
            result_dict["is_correct"] = result_dict["output_result"] == result_dict["target_result"]

            if self.metadata:
                result_dict["metadata"]["date"] = _now()
                result_dict["metadata"]["prompt_id"] = item["prompt_id"]
                result_dict["metadata"]["prompt"] = item["prompt"]
                result_dict["metadata"]["target"] = item["target"]
                result_dict["metadata"]["supervisor"] = self.supervisor.metadata()

            run_dict[item["prompt_id"]] = result_dict
            processed += 1

            if save:
                self._save_single_result(dataset_name, item["prompt_id"], run_id, result_dict)

        return processed

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

        if not self._prepared_dirs:
            self._prepare_dirs(save_dir)
            self._prepared_dirs = True

        for dataset_name, dataset_runs in self.runs.items():
            for run_id, run_dict in dataset_runs.items():
                run_dir = save_dir / dataset_name / _clean_string(run_id)
                run_dir.mkdir(exist_ok=True)
                for prompt_id, output_dict in run_dict.items():
                    file_path = (run_dir / prompt_id).with_suffix(".json")
                    output_dict["target_result"] = dict(output_dict["target_result"])
                    output_dict["output_result"] = dict(output_dict["output_result"])
                    with open(file_path, "w") as f:  # TODO : fix this
                        f.write(json.dumps(output_dict, indent=2))

    def _prepare_dirs(self, save_dir: Path | None = None):
        save_dir = save_dir or self.save_dir
        assert save_dir

        for dataset in self.datasets:
            dataset_path = save_dir / dataset.clean_name
            if self.verbose:
                print(f"Create directory {dataset_path} and all necessary parents.")
            dataset_path.mkdir(parents=True, exist_ok=True)

    def _get_result_file_path(self, dataset_name: str, prompt_id: str, run_id: str) -> Path | None:
        """Get the file path for a result given dataset_name, prompt_id and run_id."""
        if self.save_dir is None:
            return None
        run_dir = self.save_dir / dataset_name / _clean_string(run_id)
        return (run_dir / prompt_id).with_suffix(".json")

    def _load_existing_result(self, dataset_name: str, prompt_id: str, run_id: str) -> OutputDict | None:
        """Load an existing result if it exists for the given dataset, prompt_id and run_id."""
        file_path = self._get_result_file_path(dataset_name, prompt_id, run_id)
        if file_path is None or not file_path.exists():
            return None
        try:
            with open(file_path, "r") as f:
                return json.loads(f.read())
        except (json.JSONDecodeError, IOError):
            # If file is corrupted or can't be read, return None to re-process
            return None

    def _save_single_result(self, dataset_name: str, prompt_id: str, run_id: str, result_dict: OutputDict):
        """Save a single result to disk."""
        if self.save_dir is None:
            return
        file_path = self._get_result_file_path(dataset_name, prompt_id, run_id)
        if file_path is None:
            return
        file_path.parent.mkdir(parents=True, exist_ok=True)
        result_dict["target_result"] = dict(result_dict["target_result"])
        result_dict["output_result"] = dict(result_dict["output_result"])
        try:
            with open(file_path, "w") as f:
                f.write(json.dumps(result_dict, indent=2))
        except Exception:
            print(f"DEBUG: result_dict: {result_dict}")
            print(f"DEBUG: types: {[(k, type(v)) for k, v in result_dict.items()]}")
            raise


def _clean_string(string: str):
    """Clean a string such that it can be used as a name in a (Windows/UNIX) filesystem."""
    for forbidden_character in '<>:"/\\|?*':
        string = string.replace(forbidden_character, "-")
    return string
