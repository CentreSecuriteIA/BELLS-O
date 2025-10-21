"""Implement concrete class that wraps HuggingFace-accessible datasets."""

from dataclasses import dataclass, field
from typing import Any

from datasets import Dataset as HF_Dataset
from datasets import DatasetDict, load_dataset

from bells_o.datasets import Dataset


@dataclass
class HuggingFaceDataset(Dataset):
    """Wrapper class for datasets accessed through the HuggingFace `datasets` package."""

    version_name: str | None = None
    dataset_kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Load the dataset from HuggingFace and translate it to the framework."""
        if self.version_name:
            dataset = load_dataset(self.name, self.version_name, **self.dataset_kwargs)
        else:
            dataset = load_dataset(self.name, **self.dataset_kwargs)

        # check for splits
        if isinstance(dataset, DatasetDict):
            self.samples = {str(split): dataset[split].to_list() for split in dataset.keys()}
        else:  # Single split
            assert isinstance(dataset, HF_Dataset)
            self.samples = dataset.to_list()
