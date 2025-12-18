"""Implement the Anthropic (Claude) API via REST."""

from bells_o.common import ResultMapper, Usage
from bells_o.preprocessors import PreProcessing
from bells_o.supervisors.rest.auth_mappers import anthropic as auth_map
from bells_o.supervisors.rest.request_mappers import anthropic as anthropic_request_map

from ..rest_supervisor import RestSupervisor


class AnthropicSupervisor(RestSupervisor):
    """Implement the Anthropic (Claude) API via REST."""

    def __init__(
        self,
        model: str,
        usage: Usage,
        result_mapper: ResultMapper,
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
            system_prompt (str): System prompt/instruction for the model. Defaults to "".
            max_tokens (int): Maximum tokens to generate. Defaults to 1024.
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key (str | None, optional): API key to use, takes priority over `api_variable`. Defaults to None.
            api_variable (str | None, optional): Environment variable name that stores the API key. Defaults to "ANTHROPIC_API_KEY".

        """
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens

        super().__init__(
            name=model,
            usage=usage,
            res_map_fn=result_mapper,
            base_url="https://api.anthropic.com/v1/messages",
            req_map_fn=anthropic_request_map,
            auth_map_fn=auth_map,
            pre_processing=pre_processing,
            provider_name="Anthropic",
            api_key=api_key,
            api_variable=api_variable,
        )
