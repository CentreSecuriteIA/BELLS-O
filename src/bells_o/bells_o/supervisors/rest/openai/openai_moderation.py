"""Implement the pre-configured saillab/x-guard supervisor from HuggingFace."""

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing
from bells_o.resultmappers import openai_moderation as openai_moderation_result_map

from .openai import OpenAiSupervisor


class OpenAiModerationSupervisor(OpenAiSupervisor):
    """Implement the LakeraGuard supervisor via REST API, with any policy."""

    def __init__(
        self,
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "OPENAI_API_KEY",
    ):
        """Initialize the custom LakeraGuardSupervisor with a custom policy.

        Args:
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key (str | None, optional): API key to use, takes priority over `api_variable`. Defaults to None.
            api_variable (str | None, optional): Environment variable name that stores the API key. Defaults to "OPENAI_API_KEY".

        """
        usage = Usage("content_moderation")
        super().__init__(
            model="omni-moderation-latest",
            usage=usage,
            base_url="https://api.openai.com/v1/moderations",
            result_mapper=openai_moderation_result_map,
            system_prompt="",
            pre_processing=pre_processing,
            api_key=api_key,
            api_variable=api_variable,
        )
