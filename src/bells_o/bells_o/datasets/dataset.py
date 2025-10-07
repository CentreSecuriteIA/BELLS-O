"""Implement the abstract Dataset class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from bells_o.common import UsageType


@dataclass
class Dataset(ABC):
    """Abstract DataLoader class."""

    name: str
    usage: UsageType
    samples: dict[str, list] | list = field(default_factory=list, init=False, repr=False)

    @abstractmethod
    def __post_init__(self):
        """Load the dataset from e.g. HuggingFace, or local directories."""
        pass

    def __iter__(self) -> dict | list:
        return self.samples

    def __len__(self) -> int:
        if isinstance(self.samples, list):
            return len(self.samples)
        length = 0
        for ls in self.samples.values():
            length += len(ls)
        return length

    def __getitem__(self, value):
        """Return the requested data entry.

        If data contains splits but `value` is an integer,
        return the entry at position `value` if all splits were appended.

        """
        assert isinstance(value, (str, int)), "Index has to be string or int."
        # check string eligibility for list/dict
        if isinstance(value, str):
            if isinstance(self.samples, dict):
                return self.samples[value]
            raise IndexError(
                "The loaded dataset does not have any splits. Please index using integers."
            )
        # from this point on type(value) is int
        if value >= len(self):
            raise IndexError("Index out of bounds.")
        # from this point on value < len(self)

        if isinstance(self.samples, list):
            return self.samples[value]
        # from this point on type(self.sample) is dict

        # if there are splits, index with integer trough whole dataset
        split_keys = list(self.samples.keys())
        split_cuts = self._split_cuts()
        for i, cut in enumerate(split_cuts):
            if value <= cut:
                return self.samples[split_keys[i]][cut - value]

        # from this point on, value is in last split
        return self.samples[split_keys[-1]][split_cuts[-1] - value]

    def _split_lengths(self) -> list:
        """Return list of split lengths."""
        if isinstance(self.samples, list):
            return [len(self)]
        return [len(ls) for ls in self.samples.values()]

    def _split_cuts(self) -> list:
        """Return list of indeces at which the split would begin when considering the dataset as a consecutive whole.

        E.g. for split_lengths = [3,5,2], return [0,3,8].
        """
        if isinstance(self.samples, list):
            return [0]
        split_lengths = self._split_lengths()
        cuts = [0]
        for length in split_lengths:
            cuts.append(cuts[-1] + length)
        del cuts[-1]  # last cut would be same as len(self)
        return cuts
