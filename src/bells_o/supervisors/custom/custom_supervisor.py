"""Implement the base class for HuggingFace-accessible supersivor models."""

from abc import abstractmethod
from typing import Any

from bells_o.common import ResultMapper, Usage
from bells_o.preprocessors import PreProcessing

from ..supervisor import Supervisor


class CustomSupervisor(Supervisor):
    """An abstract class that builds the base for custom supervisors outside of REST or HuggingFace."""

    def __init__(
        self,
        name: str,
        usage: Usage,
        res_map_fn: ResultMapper,
        pre_processing: list[PreProcessing] = [],
        provider_name: str | None = None,
        **supervisor_kwargs,
    ):
        """Initialize the custom supervisor.

        Args:
            name (str): Name of the supervisor
            usage (Usage): The usage type of the supervisor.
            res_map_fn (ResultMapper): The `ResultMapper` used to convert results.
            pre_processing (list[PreProcessing] | None, optional): List of Preprocessing techniques for inputs. Defaults to None.
            provider_name (str | None, optional): The name of the provider of this model. Defaults to None.
            supervisor_kwargs (dict[str, Any]): The kwargs to configure the loading of the supervisor.

        """
        self._supervisor_kwargs = supervisor_kwargs

        super().__init__(name, usage, res_map_fn, pre_processing, provider_name)

        self._load_supervisor()

    @property
    def supervisor_kwargs(self) -> dict[str, Any]:  # noqa: D102
        return self._supervisor_kwargs

    def metadata(self) -> dict[str, Any]:
        """Return metadata dictionary for this Supervisor.

        Returns:
            dict: Dictionary with metadata.

        """
        metadata = super().metadata()
        if self.supervisor_kwargs is not None:
            metadata["supervisor_kwargs"] = self.supervisor_kwargs
        return metadata

    @abstractmethod
    def _load_supervisor(self):
        raise NotImplementedError("This is an abstract class.")
