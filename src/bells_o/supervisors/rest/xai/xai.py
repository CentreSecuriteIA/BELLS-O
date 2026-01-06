"""Implement the X-AI (Grok) API via REST."""

from typing import Any

from bells_o.common import ResultMapper, Usage
from bells_o.preprocessors import PreProcessing
from bells_o.supervisors.rest.auth_mappers import auth_bearer as auth_map
from bells_o.supervisors.rest.request_mappers import xai as xai_request_map

from ..rest_supervisor import RestSupervisor


class XAiSupervisor(RestSupervisor):
    """Implement the X-AI (Grok) API via REST."""

    def __init__(
        self,
        model: str,
        usage: Usage,
        result_mapper: ResultMapper,
        system_prompt: str,
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "XAI_API_KEY",
    ):
        """Initialize the XAiSupervisor.

        Args:
            model (str): The model id for the X-AI API (e.g., "grok-beta", "grok-2").
            usage (Usage): The usage type of the supervisor.
            result_mapper (ResultMapper): ResultMapper to use for this Supervisor.
            system_prompt (str): System prompt/instruction for the model.
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key (str | None, optional): API key to use, takes priority over `api_variable`. Defaults to None.
            api_variable (str | None, optional): Environment variable name that stores the API key. Defaults to "XAI_API_KEY".

        """
        self.system_prompt = system_prompt

        super().__init__(
            name=model,
            usage=usage,
            res_map_fn=result_mapper,
            base_url="https://api.x.ai/v1/chat/completions",
            req_map_fn=xai_request_map,
            auth_map_fn=auth_map,
            pre_processing=pre_processing,
            provider_name="xAI",
            api_key=api_key,
            api_variable=api_variable,
        )

    @classmethod
    def _get_token_counts(cls, output_raw: dict[str, Any]) -> dict[str, Any]:
        input_tokens = output_raw["usage"]["prompt_tokens"]
        output_tokens = output_raw["usage"]["completion_tokens"]

        return {"input_tokens": input_tokens, "output_tokens": output_tokens}
