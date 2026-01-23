"""Implement the OpenRouter API via REST."""

from typing import Any

from bells_o.common import ResultMapper, Usage
from bells_o.preprocessors import PreProcessing
from bells_o.supervisors.rest.auth_mappers import auth_bearer as auth_map
from bells_o.supervisors.rest.request_mappers import openrouter as openrouter_request_map

from ..rest_supervisor import RestSupervisor


class OpenRouterSupervisor(RestSupervisor):
    """Implement the OpenRouter API via REST."""

    def __init__(
        self,
        model: str,
        usage: Usage,
        result_mapper: ResultMapper,
        system_prompt: str,
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "OPENROUTER_API_KEY",
        reasoning_enabled: bool = False,
    ):
        """Initialize the OpenRouterSupervisor.

        Args:
            model: The model id for the OpenRouter API (e.g., "openai/gpt-oss-safeguard-20b").
            usage: The usage type of the supervisor.
            result_mapper: ResultMapper to use for this Supervisor.
            system_prompt: System prompt/instruction for the model.
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key: API key to use, takes priority over `api_variable`. Defaults to None.
            api_variable: Environment variable name that stores the API key. Defaults to "OPENROUTER_API_KEY".
            reasoning_enabled: Whether to enable reasoning for models that support it. Defaults to False.

        """
        self.system_prompt = system_prompt
        self.reasoning_enabled = reasoning_enabled

        super().__init__(
            name=model,
            usage=usage,
            res_map_fn=result_mapper,
            base_url="https://openrouter.ai/api/v1/chat/completions",
            req_map_fn=openrouter_request_map,
            auth_map_fn=auth_map,
            pre_processing=pre_processing,
            provider_name="OpenRouter",
            api_key=api_key,
            api_variable=api_variable,
        )

    @classmethod
    def _get_token_counts(cls, output_raw: dict[str, Any]) -> dict[str, Any]:
        try:
            input_tokens = output_raw["usage"]["prompt_tokens"]
            output_tokens = output_raw["usage"]["completion_tokens"]
        except KeyError as e:
            print("DEBUGGING: output_raw dict:")
            print(output_raw)
            raise KeyError from e
            
        return {"input_tokens": input_tokens, "output_tokens": output_tokens}
