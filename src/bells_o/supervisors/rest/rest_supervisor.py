"""Implement the base class for REST-accessible supersivors."""

from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from os import getenv
from time import time
from typing import Any

from requests import post

from bells_o.common import AuthMapper, OutputDict, RequestMapper, ResultMapper, Usage
from bells_o.preprocessors import PreProcessing

from ..supervisor import Supervisor


@dataclass(kw_only=True)
class RestSupervisor(Supervisor):
    """A concrete class that enables access to supervisors via REST API."""

    # TODO: doc string
    def __init__(
        self,
        name: str,
        usage: Usage,
        res_map_fn: ResultMapper,
        base_url: str,
        req_map_fn: RequestMapper,
        auth_map_fn: AuthMapper,
        pre_processing: list[PreProcessing] = [],
        provider_name: str | None = None,
        api_key: str | None = None,
        api_variable: str | None = None,
        needs_api: bool = True,
        rate_limit_code: int = 429,
        custom_header: dict[str, str] = {},
    ):
        assert not needs_api or self.api_key, (
            "You have to specify either the environment variabe in which the API key can be found (`api_variable`), or the API key itself (`api_key`)."
        )
        super().__init__(name, usage, res_map_fn, pre_processing)

        self.base_url = base_url
        self.req_map_fn = req_map_fn
        self.auth_map_fn = auth_map_fn
        self.pre_processing = pre_processing
        self._provider_name = provider_name  # private
        self._api_key = api_key
        self._api_variable = api_variable
        self._needs_api = needs_api
        self.rate_limit_code = rate_limit_code
        self.custom_header = custom_header

    @property
    def api_key(self) -> str:  # noqa: D102
        if not self._needs_api:
            return ""
        return self._api_key or getenv(self.api_variable, "")

    @api_key.setter
    def api_key(self, value: str):
        self._api_key = value
        self._needs_api = True

    @property
    def api_variable(self) -> str:  # noqa: D102
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

    @classmethod
    @abstractmethod
    def _get_token_counts(cls, output_raw: dict[str, Any]) -> dict[str, Any]:
        input_tokens = output_raw["some_key"]
        output_tokens = output_raw["some_other_key"]

        return {"input_tokens": input_tokens, "output_tokens": output_tokens}

    def _judge_sample(
        self,
        prompt: str,
    ) -> OutputDict:
        """Run an individual POST request for inference.

        Args:
            prompt (str): The prompt string to check.
            output_type (Literal["output_dict", "request"]): The type of the return value. Returns an `OutputDict` if `"output_dict"`,
                and returns `tuple[Response, float]` if `"request"`.


        Returns:
            OutputDict | tuple[Response, float]: The output of the Supervisor and corresponding metadata, mapped to an OutputDict or as a Response object.

        """
        tried_once = False  # to distinguish between trying and retrying information
        no_valid_response = True  # to manage retries

        while no_valid_response:
            if tried_once:
                print("INFO: Retrying generation in 5s. Hit rate limit.")
            else:
                print("INFO: Generating judgement.")

            start_time = time()
            headers = self.auth_map_fn(self) | self.custom_header
            response = post(
                self.base_url,
                json=self.req_map_fn(self, prompt),
                headers=headers,
            )
            generation_time = time() - start_time

            tried_once = True
            no_valid_response = response.status_code == self.rate_limit_code

        output_raw = response.json()
        metadata = self._get_token_counts(output_raw)
        metadata["latency"] = generation_time

        return OutputDict(output_raw=output_raw, metadata=metadata)

    def judge(self, prompts: list[str] | str) -> list[OutputDict]:
        """Evaluate a (batch of) prompt(s simultaneously).

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
                    lambda prompt: self._judge_sample(prompt),
                    prompts,
                )
            )

        return outputs
