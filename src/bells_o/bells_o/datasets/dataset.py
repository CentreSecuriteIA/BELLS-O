"""Implement the abstract Dataset class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, overload

from bells_o.common import Usage


@dataclass
class Dataset(ABC):
    """Abstract DataLoader class."""

    name: str
    usage: Usage
    target_map_fn: Callable | None = field(default=None)
    filters: dict[str, list[Any]] | None = field(default_factory=dict)
    samples: dict[str, list[dict[str, str]]] | list[dict[str, str]] = field(
        default_factory=list, init=False, repr=False
    )

    @abstractmethod
    def __post_init__(self):
        """Load the dataset from e.g. HuggingFace, or local directories. Must enable filtering."""
        self.clean_name = _clean_string(self.name)
        _add_prompt_id(self)
        pass

    # TODO Implement metadata

    def splits(self) -> list[str]:
        """Return a list of the splits."""
        if isinstance(self.samples, dict):
            return list(self.samples.keys())
        return []

    def filter(self, filters: dict[str, list[Any]] | None = None):
        """Filter a list in-place for given filters.

        Filters are of shape {attribute: [allowed values]}.
        """

        def _filter_list(l: list[dict[str, str]], filters: dict[str, list[Any]]):
            l[:] = [
                sample
                for sample in l
                if all(sample[attribute] in values for attribute, values in filters.items())
            ]

        filt = filters or self.filters
        if filt is not None:
            if isinstance(self.samples, dict):
                for split in self.splits():
                    _filter_list(self.samples[split], filt)
            else:
                _filter_list(self.samples, filt)

    def _split_lengths(self) -> list:
        """Return list of split lengths."""
        if isinstance(self.samples, list):
            return [len(self)]
        return [len(ls) for ls in self.samples.values()]

    def _split_starts(self) -> list:
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

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self) -> int:
        if isinstance(self.samples, list):
            return len(self.samples)
        length = 0
        for ls in self.samples.values():
            length += len(ls)
        return length

    @overload
    def __getitem__(self, value: str | slice) -> list[dict[str, str]]: ...

    @overload
    def __getitem__(self, value: int) -> dict[str, str]: ...

    def __getitem__(self, value: str | int | slice) -> list[dict[str, str]] | dict[str, str]:
        """Return the requested data entry.

        If data contains splits but `value` is an integer,
        return the entry at position `value` if all splits were appended.

        """
        assert isinstance(value, (str, int, slice)), "Index has to be string int, or slice."
        # check string eligibility for list/dict
        if isinstance(value, str):
            if isinstance(self.samples, dict):
                return self.samples[value]
            raise IndexError(
                "The loaded dataset does not have any splits. Please index using integers."
            )
        # from this point on value is int or slice
        if isinstance(self.samples, list):
            # can index or slice directly
            return self.samples[value]

        # from this point on self.samples is dict
        if isinstance(value, int):
            while value < 0:  # negative index case
                value += len(self)
            if not (0 <= value < len(self)):
                raise IndexError("Index out of bounds.")

            # index with integer trough all splits
            split_keys = list(self.samples.keys())
            split_starts = self._split_starts()
            for i, start in enumerate(split_starts):
                if value < start:
                    # if the current start is beyond the value,
                    # then return value offset by the previous start of the previous list.
                    return self.samples[split_keys[i - 1]][value - split_starts[i - 1]]

            # from this point on, value is in last split
            return self.samples[split_keys[-1]][split_starts[-1] - value]

        if isinstance(value, slice):
            # translate slice into indices
            start, stop, step = value.indices(len(self))
            indices = list(range(start, stop, step))

            # use self[int] from slice indices
            return [self[index] for index in indices]


def _clean_string(string: str):
    """Clean a string such that it can be used as a name in a (Windows/UNIX) filesystem."""
    for forbidden_character in '<>:"/\\|?*':
        string = string.replace(forbidden_character, "-")
    return string


def _add_prompt_id(obj: Dataset):
    """Add prompt ids to dataset.

    Prompt IDs are of shape `dataset.name`_`index`, where the dataset name is cleaned for file name compliance.
    """
    for i, sample in enumerate(obj):
        sample["prompt_id"] = f"{obj.clean_name}_{i}"
