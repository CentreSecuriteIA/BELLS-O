"""Implement the Anthropic (Claude) API via REST."""

from functools import partial
from typing import Self, cast

from bells_o.common import AuthMapper, RequestMapper, ResultMapper, Usage
from bells_o.preprocessors import PreProcessing

from bells_o.supervisors.rest.auth_mappers import anthropic as auth_map
from bells_o.supervisors.rest.request_mappers import anthropic as anthropic_request_map

from ..custom_endpoint import RestSupervisor


class AnthropicSupervisor(RestSupervisor):
    """Implement the Anthropic (Claude) API via REST."""

    def __init__(
        self,
        model: str,
        usage: Usage,
        result_mapper: ResultMapper,
        base_url: str = "https://api.anthropic.com/v1/messages",
        system_prompt: str = "",
        max_tokens: int = 1024,
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "ANTHROPIC_API_KEY",
    ):
        """Initialize the AnthropicSupervisor.

        Args:
            model (str): The model id for the Anthropic API (e.g., "claude-3-5-sonnet-20241022", "claude-3-opus-20240229").
            usage (Usage): The usage type of the supervisor.
            result_mapper (ResultMapper): ResultMapper to use for this Supervisor.
            base_url (str): Base URL of the API endpoint. Defaults to "https://api.anthropic.com/v1/messages".
            system_prompt (str): System prompt/instruction for the model. Defaults to "".
            max_tokens (int): Maximum tokens to generate. Defaults to 1024.
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key (str | None, optional): API key to use, takes priority over `api_variable`. Defaults to None.
            api_variable (str | None, optional): Environment variable name that stores the API key. Defaults to "ANTHROPIC_API_KEY".

        """
        self.name: str = model
        self.provider_name: str | None = "Anthropic"
        self.custom_header = {}
        self.base_url: str = base_url
        self.usage: Usage = usage
        self.res_map_fn: ResultMapper = cast(ResultMapper, partial(result_mapper, usage=self.usage))
        self.req_map_fn: RequestMapper[Self] = anthropic_request_map
        self.auth_map_fn: AuthMapper = auth_map
        self.pre_processing = pre_processing
        self.api_key = api_key
        self.api_variable = api_variable
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

        super().__post_init__()

