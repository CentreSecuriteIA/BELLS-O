"""Implement the VirtueGuard Text Lite Moderation Supervisor via Together AI."""

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import virtue_guard as virtue_guard_result_map

from .together import TogetherAISupervisor


class VirtueGuardTextLiteModerationSupervisor(TogetherAISupervisor):
    """Together AI supervisor for VirtueGuard Text Lite model configured for direct moderation.

    Uses Together AI's VirtueGuard Text Lite model (Virtue-AI/VirtueGuard-Text-Lite) for moderation 
    without a system prompt. The prompt is sent directly as user content for the model to classify.
    """

    def __init__(
        self,
        model: str = "Virtue-AI/VirtueGuard-Text-Lite",
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "TOGETHER_API_KEY",
    ):
        """Initialize the VirtueGuardTextLiteModerationSupervisor.

        Args:
            model: Together AI model id. Defaults to "Virtue-AI/VirtueGuard-Text-Lite".
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key: Together AI API key (if given, overrides env). Defaults to None.
            api_variable: Env var name for the API key. Defaults to "TOGETHER_API_KEY".
        """
        super().__init__(
            model=model,
            usage=Usage("content_moderation"),
            result_mapper=virtue_guard_result_map,
            system_prompt="",  # No system prompt for moderation mode
            pre_processing=pre_processing,
            api_key=api_key,
            api_variable=api_variable,
        )

