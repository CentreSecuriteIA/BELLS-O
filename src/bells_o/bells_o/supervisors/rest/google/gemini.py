"""Implement the Google AI Studio (Gemini) API via REST."""

from functools import partial
from typing import Self, cast

from bells_o.common import AuthMapper, RequestMapper, ResultMapper, Usage
from bells_o.preprocessors import PreProcessing

from bells_o.supervisors.rest.auth_mappers import google_api_key as auth_map
from bells_o.supervisors.rest.request_mappers import google as google_request_map

from ..custom_endpoint import RestSupervisor


class GeminiSupervisor(RestSupervisor):
    """Implement the Google AI Studio (Gemini) API via REST.

    Base supervisor for Gemini models without safety settings.
    Uses the Google AI Studio API (generativelanguage.googleapis.com) with API key authentication.
    For moderation use cases with safety settings, use VertexModerationSupervisor.
    """

    def __init__(
        self,
        model: str,
        usage: Usage,
        result_mapper: ResultMapper,
        system_prompt: str = "",
        safety_settings: list[dict] | None = None,
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "GEMINI_API_KEY",
    ):
        """Initialize GeminiSupervisor.

        Args:
            model: Gemini model id, e.g. "gemini-1.5-pro" or "gemini-2.5-flash".
            usage: Usage type for BELLS-O.
            result_mapper: ResultMapper for this Supervisor.
            system_prompt: System-level instruction → mapped to `system_instruction`.
            safety_settings: Optional list of safetySettings dicts (category + threshold).
                Defaults to None (no safety settings). For moderation use cases, use VertexModerationSupervisor.
            pre_processing: PreProcessing steps.
            api_key: Google AI Studio API key (if given, overrides env).
            api_variable: Env var name for the API key. Defaults to "GEMINI_API_KEY".
        """
        self.name: str = model
        self.provider_name: str | None = "GoogleAIStudio"
        self.custom_header = {}

        # Google AI Studio generateContent endpoint
        self.base_url: str = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"
        )

        self.usage: Usage = usage
        self.res_map_fn: ResultMapper = cast(ResultMapper, partial(result_mapper, usage=self.usage))
        self.req_map_fn: RequestMapper[Self] = google_request_map
        self.auth_map_fn: AuthMapper = auth_map  # x-goog-api-key header
        self.pre_processing = pre_processing
        self.api_key = api_key
        self.api_variable = api_variable

        # Gemini-specific settings
        self.system_prompt = system_prompt
        self.safety_settings = safety_settings or []

        super().__post_init__()

