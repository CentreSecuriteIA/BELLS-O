"""Implement the Llama Guard 4B Moderation Supervisor via Together AI."""

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import openai_compatible_unsafe as llama_guard_4_rest_result_map

from .together import TogetherAISupervisor


class LlamaGuard4BModerationSupervisor(TogetherAISupervisor):
    """Together AI supervisor for Llama Guard 4B model configured for direct moderation.

    Uses Together AI's Llama Guard 4B model (meta-llama/Llama-Guard-4-12B) for moderation
    without a system prompt. The prompt is sent directly as user content for the model to classify.
    """

    def __init__(
        self,
        model: str = "meta-llama/Llama-Guard-4-12B",
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "TOGETHER_API_KEY",
    ):
        """Initialize the LlamaGuard4BModerationSupervisor.

        Args:
            model: Together AI model id. Defaults to "meta-llama/Llama-Guard-4-12B".
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key: Together AI API key (if given, overrides env). Defaults to None.
            api_variable: Env var name for the API key. Defaults to "TOGETHER_API_KEY".

        """
        super().__init__(
            model=model,
            usage=Usage("content_moderation"),
            result_mapper=llama_guard_4_rest_result_map,
            system_prompt="",  # No system prompt for moderation mode
            pre_processing=pre_processing,
            api_key=api_key,
            api_variable=api_variable,
        )
