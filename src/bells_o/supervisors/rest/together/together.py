"""Implement the Together AI API via REST."""

from functools import partial
from typing import Self, cast

from bells_o.common import AuthMapper, RequestMapper, ResultMapper, Usage
from bells_o.preprocessors import PreProcessing
from bells_o.supervisors.rest.auth_mappers import auth_bearer as auth_map
from bells_o.supervisors.rest.request_mappers import together as together_request_map

from ..rest_supervisor import RestSupervisor


class TogetherAISupervisor(RestSupervisor):
    """Implement the Together AI API via REST."""

    def __init__(
        self,
        model: str,
        usage: Usage,
        result_mapper: ResultMapper,
        base_url: str = "https://api.together.xyz/v1/chat/completions",
        system_prompt: str = "",
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "TOGETHER_API_KEY",
    ):
        """Initialize the TogetherAISupervisor.

        Args:
            model (str): The model id for the Together AI API (e.g., "openai/gpt-oss-20b", "meta-llama/Llama-Guard-4-12B").
            usage (Usage): The usage type of the supervisor.
            result_mapper (ResultMapper): ResultMapper to use for this Supervisor.
            base_url (str): Base URL of the API endpoint. Defaults to "https://api.together.xyz/v1/chat/completions".
            system_prompt (str): System prompt/instruction for the model. Defaults to "".
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key (str | None, optional): API key to use, takes priority over `api_variable`. Defaults to None.
            api_variable (str | None, optional): Environment variable name that stores the API key. Defaults to "TOGETHER_API_KEY".

        """
        self.name: str = model
        self.provider_name: str | None = "Together AI"
        self.custom_header = {}
        self.base_url: str = base_url
        self.usage: Usage = usage
        self.res_map_fn: ResultMapper = cast(ResultMapper, partial(result_mapper, usage=self.usage))
        self.req_map_fn: RequestMapper[Self] = together_request_map
        self.auth_map_fn: AuthMapper = auth_map
        self.pre_processing = pre_processing
        self.api_key = api_key
        self.api_variable = api_variable
        self.system_prompt = system_prompt

        super().__post_init__()
