"""Implement the OpenAI API via REST."""

from functools import partial
from typing import Self, cast

from bells_o.common import AuthMapper, RequestMapper, ResultMapper, Usage
from bells_o.preprocessors import PreProcessing

# from bells_o.resultmappers import generalist as generalist_result_map
from bells_o.supervisors.rest.auth_mappers import auth_bearer as auth_map
from bells_o.supervisors.rest.request_mappers import openai as openai_request_map

from ..custom_endpoint import RestSupervisor


# TODO: Add generalist mapper as default
class OpenAiSupervisor(RestSupervisor):
    """Implement the OpenAI API via REST."""

    def __init__(
        self,
        model: str,
        usage: Usage,
        result_mapper: ResultMapper,  # = generalist_result_map,
        base_url: str = "https://api.openai.com/v1/responses",
        system_prompt: str = "",  # TODO: add generalist system prompt
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "OPENAI_API_KEY",
    ):
        """Initialize the custom OpenAiSupervisor with a custom system prompt.

        This implementation needs a specified `usage_type`. The system prompt defines the

        Args:
            model (str): The model id for the OpenAI API.
            usage (Usage): The usage of the supervisor, defined by the passed `system_prompt`.
            result_mapper (ResultMapper): ResultMapper to use for this Supervisor.
            system_prompt (str): A string that describes how to classify text. Defaults to the generalist supervisor prompt. # TODO: add generalist supervisor prompt.
            base_url (str): Base URL of the API endpoint to use. Defaults to "https://api.openai.com/v1/responses".
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key (str | None, optional): API key to use, takes priority over `api_variable`. Defaults to None.
            api_variable (str | None, optional): Environment variable name that stores the API key. Defaults to "OPENAI_API_KEY".

        """
        self.name: str = model
        self.provider_name: str | None = "OpenAI"
        self.base_url: str = base_url
        self.usage: Usage = usage
        self.res_map_fn: ResultMapper = cast(ResultMapper, partial(result_mapper, usage=self.usage))
        self.req_map_fn: RequestMapper[Self] = openai_request_map
        self.auth_map_fn: AuthMapper = auth_map
        self.pre_processing = pre_processing
        self.api_key = api_key
        self.api_variable = api_variable
        self.system_prompt = system_prompt

        super().__post_init__()
