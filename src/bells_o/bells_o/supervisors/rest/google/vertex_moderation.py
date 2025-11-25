"""Implement the Vertex AI Moderation API via REST."""

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import vertex_moderation as vertex_moderation_result_map

from .gemini import GeminiSupervisor

# All the harmful categories that are defined by Vertex AI Moderation.
DEFAULT_SAFETY_SETTINGS = [
    {
        "category": "HARM_CATEGORY_HATE_SPEECH",
        "threshold": "BLOCK_HIGH_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_HARASSMENT",
        "threshold": "BLOCK_HIGH_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SEXUALLY_EXPLICIT",
        "threshold": "BLOCK_HIGH_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
        "threshold": "BLOCK_HIGH_AND_ABOVE",
    },
    {
        "category": "HARM_CATEGORY_SELF_HARM",
        "threshold": "BLOCK_HIGH_AND_ABOVE",
    },
]


class VertexModerationSupervisor(GeminiSupervisor):
    """Content-moderation style supervisor using Vertex Gemini + safetySettings.

    Extends GeminiSupervisor with default safety settings for content moderation.
    """

    def __init__(
        self,
        project_id: str,
        location: str = "europe-west1",
        model: str = "gemini-1.5-pro",
        pre_processing: list[PreProcessing] = [],
        safety_settings: list[dict] | None = None,
        system_prompt: str = "",
        api_key: str | None = None,
        api_variable: str = "VERTEX_ACCESS_TOKEN",
    ):
        """Initialize the VertexModerationSupervisor.

        Args:
            project_id: GCP project id.
            location: GCP location (e.g. "europe-west1", "us-central1"). Defaults to "europe-west1".
            model: Gemini model id. Defaults to "gemini-1.5-pro".
            pre_processing: List of PreProcessing steps to apply to prompts. Defaults to [].
            safety_settings: List of Vertex safetySettings dicts. Defaults to DEFAULT_SAFETY_SETTINGS.
            system_prompt: System-level instruction. Defaults to "".
            api_key: OAuth2 bearer token (if given, overrides env). Defaults to None.
            api_variable: Env var name for the token. Defaults to "VERTEX_ACCESS_TOKEN".
        """
        super().__init__(
            model=model,
            usage=Usage("content_moderation"),
            result_mapper=vertex_moderation_result_map,
            project_id=project_id,
            location=location,
            system_prompt=system_prompt,
            safety_settings=safety_settings or DEFAULT_SAFETY_SETTINGS,
            pre_processing=pre_processing,
            api_key=api_key,
            api_variable=api_variable,
        )

