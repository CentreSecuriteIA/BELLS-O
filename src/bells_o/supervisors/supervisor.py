"""Implements the abstract Supervisor Class."""

from abc import ABC, abstractmethod, property
from functools import partial
from typing import Any

from bells_o.common import OutputDict, ResultMapper, Usage
from bells_o.preprocessors import PreProcessing


class Supervisor(ABC):
    """Abstract base class for Supervisors.

    Attributes:
        name (str): Name of the supervisor.
        res_map_fn (Callable): Function to map the output to a `Result` dict.
        pre_processing (Optional[List[PreProcessing]]): List of PreProcessing techniques that should be applied.

    """

    def __init__(self, name: str, usage: Usage, res_map_fn: ResultMapper, pre_processing: list[PreProcessing] = []):
        """Initialize the supervisor.

        Args:
            name (str): Name of the supervisor
            usage (Usage): The usage type of the supervisor.
            res_map_fn (ResultMapper): The `ResultMapper` used to convert results.
            pre_processing (list[PreProcessing] | None, optional): List of Preprocessing techniques for inputs. Defaults to None.

        """
        self._name = name
        self._usage = Usage
        self._res_map_fn = res_map_fn
        self.pre_processing = pre_processing

    @property
    def name(self) -> str:  # noqa: D102
        return self._name

    @property
    def usage(self) -> str:  # noqa: D102
        return self._usage

    def __repr__(self) -> str:
        """Represent class object as string."""
        return f'{self.__name__}("{self.name}", "{self.usage}")'

    def __call__(self, inputs: str | list[str], *args, **kwargs) -> list[OutputDict]:
        """Complete full judging process."""
        if not isinstance(inputs, list):
            inputs = [inputs]

        inputs = self.pre_process(inputs)

        outputs: list[OutputDict] = self.judge(inputs)
        for output in outputs:
            # Check if res_map_fn is a partial function (usage already bound)
            if isinstance(self.res_map_fn, partial):
                output["output_result"] = self.res_map_fn(output["output_raw"])  # pyright: ignore[reportArgumentType]
            else:
                output["output_result"] = self.res_map_fn(output["output_raw"], self.usage)  # pyright: ignore[reportArgumentType]
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

    def pre_process(self, inputs):
        """Apply all preprocessing steps.

        Concrete classes will likely need a tokenization equivalent implemented.
        """
        if self.pre_processing:
            for pre_processor in self.pre_processing:
                inputs = [pre_processor(input) for input in inputs]
        return inputs
