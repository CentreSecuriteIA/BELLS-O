"""Implement the base class for REST-accessible supersivors."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from os import getenv
from time import time

from requests import post

from bells_o.common import JsonMapper, OutputDict

from ..supervisor import Supervisor


@dataclass(kw_only=True)
class RestSupervisor(Supervisor):
    """A concrete class that enables access to supervisors via REST API."""

    base_url: str
    req_map_fn: JsonMapper
    auth_map_fn: JsonMapper  # type: ignore
    api_key: str | None = None  # type: ignore
    api_variable: str | None = None
    provider_name: str | None = None
    model_name: str | None = None

    def __post_init__(self):
        """Load the model and tokenizer from HuggingFace."""
        self._api_variable = self.api_variable
        assert self.api_key, (
            "You have to specify either the environment variabe in which the API key can be found (`api_variable`), or the API key itself (`api_key`)."
        )
        del self.api_variable
        super().__post_init__()

    @property
    def api_key(self):  # noqa: F811
        """Return the API key set for this supervisor."""
        return self._api_key or getenv(self._api_variable)  # type: ignore

    @api_key.setter
    def api_key(self, value: str | None):
        self._api_key = value

    def metadata(self, return_date: bool = False) -> dict[str, str]:
        """Return metadata dictionary for this Supervisor.

        Args:
            return_date (bool): If the current time should be returned.

        Returns:
            dict: Dictionary with metadata.

        """
        metadata = {
            "provider": getattr(self, "provider_name", "Unknown"),
            "model": getattr(self, "model_name", "Unknown"),
            "usage": repr(self.usage),
        }

        if return_date:
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            metadata["date"] = date

        return metadata

    def _judge_sample(self, prompt: str) -> OutputDict:
        """Run an individual POST request for inference.

        Args:
            prompt (str): The prompt string to check.

        Returns:
            OutputDict: The output of the Supervisor and corresponding metadata.

        """
        start_time = time()
        response = post(
            self.base_url, json=self.req_map_fn(prompt, self), headers=self.auth_map_fn(self)
        )
        generation_time = time() - start_time

        return OutputDict(raw_result=response.json(), metadata={"latency": generation_time})

    def judge(self, prompts: list[str] | str) -> list[OutputDict]:
        """Evaluate a batch of prompts simultaneously.

        Args:
            prompts: List of prompts.

        Returns:
            list[OutputDict]: List of outputs and metadata.

        """
        if not prompts:
            return []

        if not isinstance(prompts, list):
            prompts = [prompts]

        with ThreadPoolExecutor() as executor:
            # the next statement runs a request for every prompt in parallel
            outputs = list(
                executor.map(
                    lambda prompt: self._judge_sample(
                        prompt,
                    ),
                    prompts,
                )
            )

        return outputs
