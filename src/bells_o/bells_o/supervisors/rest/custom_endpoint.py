"""Implement the base class for REST-accessible supersivors."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import getenv
from time import time
from typing import Any

from requests import post

from bells_o.common import AuthMapper, OutputDict, RequestMapper

from ..supervisor import Supervisor


@dataclass(kw_only=True)
class RestSupervisor(Supervisor):
    """A concrete class that enables access to supervisors via REST API."""

    base_url: str
    req_map_fn: RequestMapper
    auth_map_fn: AuthMapper  # type: ignore
    api_key: str | None = None  # type: ignore
    api_variable: str | None = None  # type: ignore
    provider_name: str | None = None
    needs_api: bool = True

    def __post_init__(self):
        """Load the model and tokenizer from HuggingFace."""
        assert not self.needs_api or self.api_key, (
            "You have to specify either the environment variabe in which the API key can be found (`api_variable`), or the API key itself (`api_key`)."
        )
        super().__post_init__()

    @property
    def api_key(self) -> str:
        """Return the API key set for this supervisor."""
        return self._api_key or getenv(self.api_variable, "")

    @api_key.setter
    def api_key(self, value: str | None):
        self._api_key = value

    @property
    def api_variable(self) -> str:  # noqa: F811
        """Return the environment variable set for the API key set for this supervisor."""
        return self._api_variable if self._api_variable is not None else ""

    @api_variable.setter
    def api_variable(self, value: str | None):
        self._api_variable = value

    def metadata(self) -> dict[str, Any]:
        """Return metadata dictionary for this Supervisor.

        Returns:
            dict: Dictionary with metadata.

        """
        metadata = super().metadata()
        metadata["url"] = self.base_url
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

        return OutputDict(output_raw=response.json(), metadata={"latency": generation_time})

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
