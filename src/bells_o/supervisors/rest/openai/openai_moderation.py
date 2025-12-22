"""Implement the OpenAI Moderation API via REST."""

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import openai_moderation as openai_moderation_result_map

from .openai import OpenAiSupervisor


class OpenAiModerationSupervisor(OpenAiSupervisor):
    """Implement the OpenAI Moderation API via REST."""

    def __init__(
        self,
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "OPENAI_API_KEY",
    ):
        """Initialize the OpenAiModerationSupervisor.

        Args:
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key (str | None, optional): API key to use, takes priority over `api_variable`. Defaults to None.
            api_variable (str | None, optional): Environment variable name that stores the API key. Defaults to "OPENAI_API_KEY".

        """
        super().__init__(
            model="omni-moderation-latest",
            usage=Usage("content_moderation"),
            base_url="https://api.openai.com/v1/moderations",
            result_mapper=openai_moderation_result_map,
            system_prompt="",
            pre_processing=pre_processing,
            api_key=api_key,
            api_variable=api_variable,
        )
