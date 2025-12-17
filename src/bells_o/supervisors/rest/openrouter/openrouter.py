"""Implement the OpenRouter API via REST."""

from functools import partial
from typing import Self, cast

from bells_o.common import AuthMapper, RequestMapper, ResultMapper, Usage
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
        base_url: str = "https://openrouter.ai/api/v1/chat/completions",
        system_prompt: str = "",
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
            base_url: Base URL of the API endpoint. Defaults to "https://openrouter.ai/api/v1/chat/completions".
            system_prompt: System prompt/instruction for the model. Defaults to "".
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key: API key to use, takes priority over `api_variable`. Defaults to None.
            api_variable: Environment variable name that stores the API key. Defaults to "OPENROUTER_API_KEY".
            reasoning_enabled: Whether to enable reasoning for models that support it. Defaults to False.

        """
        self.name: str = model
        self.provider_name: str | None = "OpenRouter"
        self.custom_header = {}
        self.base_url: str = base_url
        self.usage: Usage = usage
        self.res_map_fn: ResultMapper = cast(ResultMapper, partial(result_mapper, usage=self.usage))
        self.req_map_fn: RequestMapper[Self] = openrouter_request_map
        self.auth_map_fn: AuthMapper = auth_map
        self.pre_processing = pre_processing
        self.api_key = api_key
        self.api_variable = api_variable
        self.system_prompt = system_prompt
        self.reasoning_enabled = reasoning_enabled

        super().__post_init__()
