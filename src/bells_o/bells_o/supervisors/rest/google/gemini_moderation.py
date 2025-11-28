"""Implement the Vertex AI Moderation API via REST."""

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import gemini_moderation as gemini_moderation_result_map

from .gemini import GeminiSupervisor

# All the harmful categories that are defined by default
DEFAULT_SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_LOW_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_LOW_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_LOW_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_LOW_AND_ABOVE",
    }
]


class GeminiModerationSupervisor(GeminiSupervisor):
    """Content-moderation style supervisor using Gemini + safetySettings.

    Extends GeminiSupervisor with default safety settings for content moderation.
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        pre_processing: list[PreProcessing] = [],
        safety_settings: list[dict] | None = None,
        system_prompt: str = "",
        api_key: str | None = None,
        api_variable: str = "GEMINI_API_KEY",
    ):
        """Initialize the GeminiModerationSupervisor.

        Args:
            model: Gemini model id. Defaults to "gemini-2.5-flash".
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            safety_settings: List of safetySettings dicts. Defaults to DEFAULT_SAFETY_SETTINGS.
            system_prompt: System-level instruction. Defaults to "".
            api_key: Google AI Studio API key (if given, overrides env). Defaults to None.
            api_variable: Env var name for the API key. Defaults to "GEMINI_API_KEY".
        """
        super().__init__(
            model=model,
            usage=Usage("content_moderation"),
            result_mapper=gemini_moderation_result_map,
            system_prompt=system_prompt,
            safety_settings=DEFAULT_SAFETY_SETTINGS,
            pre_processing=pre_processing,
            api_key=api_key,
            api_variable=api_variable,
        )

