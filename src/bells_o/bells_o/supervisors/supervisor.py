"""Implements the abstract Supervisor Class."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

from bells_o.common import Mapper, Result, Usage
from bells_o.preprocessors import PreProcessing


# Define an abstract base class
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
    res_map_fn: Mapper
    pre_processing: list[PreProcessing] | None

    @abstractmethod
    def __post_init__(self):
        """Set up the rest of the supervisor. E.g. load the model from HuggingFace."""
        self.pre_processing = self.pre_processing or []

    def __call__(self, input, *args, **kwargs) -> Result:
        """Complete full judging process."""
        input = self.pre_process(input)
        output = self.judge(input)
        result = self.res_map_fn(output)
        return result

    @abstractmethod
    def judge(self, *args, **kwargs) -> str:
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
