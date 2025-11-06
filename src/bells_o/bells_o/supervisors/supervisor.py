"""Implements the abstract Supervisor Class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from bells_o.common import OutputDict, ResultMapper, Usage
from bells_o.preprocessors import PreProcessing


# Define an abstract base class
# TODO: get rid of dataclass
@dataclass
class Supervisor(ABC):
    """Abstract base class for Supervisors.

    Attributes:
        name (str): Name of the supervisor.
        res_map_fn (Callable): Function to map the output to a `Result` dict.
        pre_processing (Optional[List[PreProcessing]]): List of PreProcessing techniques that should be applied.

    """

    name: str
    usage: Usage
    res_map_fn: ResultMapper
    pre_processing: list[PreProcessing] | None

    @abstractmethod
    def __post_init__(self):
        """Set up the rest of the supervisor. E.g. load the model from HuggingFace."""
        self.pre_processing = self.pre_processing or []

    def __call__(self, inputs: str | list[str], *args, **kwargs) -> list[OutputDict]:
        """Complete full judging process."""
        if not isinstance(inputs, list):
            inputs = [inputs]
        inputs = [self.pre_process(input) for input in inputs]
        outputs: list[OutputDict] = self.judge(inputs)
        for output in outputs:
            output["output_result"] = self.res_map_fn(output["output_raw"])  # pyright: ignore[reportArgumentType]
        return outputs

    def metadata(self) -> dict[str, Any]:
        """Return metadata dictionary for this Supervisor.

        Returns:
            dict: Dictionary with metadata.

        """
        metadata = {
            "provider": getattr(self, "provider_name", "Unknown"),
            "model": getattr(self, "name", "Unknown"),
            "usage": repr(self.usage),
            "supervisor_type": self.__class__.__name__,
        }

        return metadata

    @abstractmethod
    def judge(self, *args, **kwargs) -> list[OutputDict]:
        """Run one evaluation on the supervisor.

        Similar to `forward` in PyTorch, it expects prepped inputs s.t.
        they can be used directly with the supervisor.
        """
        pass

    def pre_process(self, string):
        """Apply all preprocessing steps.

        Concrete classes will likely need a tokenization equivalent implemented.
        """
        if self.pre_processing:
            for pre_processor in self.pre_processing:
                string = pre_processor(string)
        return string
