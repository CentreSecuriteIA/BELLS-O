"""Implement the Vertex AI (Gemini) API via REST."""

from functools import partial
from typing import Self, cast

from bells_o.common import AuthMapper, RequestMapper, ResultMapper, Usage
from bells_o.preprocessors import PreProcessing

from bells_o.supervisors.rest.auth_mappers import auth_bearer as auth_map
from bells_o.supervisors.rest.request_mappers import vertex as vertex_request_map

from ..custom_endpoint import RestSupervisor


class GeminiSupervisor(RestSupervisor):
    """Implement the Vertex AI (Gemini) API via REST.

    Base supervisor for Gemini models without safety settings.
    For moderation use cases with safety settings, use VertexModerationSupervisor.
    """

    def __init__(
        self,
        model: str,
        usage: Usage,
        result_mapper: ResultMapper,
        project_id: str,
        location: str = "europe-west1",
        system_prompt: str = "",
        safety_settings: list[dict] | None = None,
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "VERTEX_ACCESS_TOKEN",
    ):
        """Initialize GeminiSupervisor.

        Args:
            model: Gemini model id, e.g. "gemini-1.5-pro".
            usage: Usage type for BELLS-O.
            result_mapper: ResultMapper for this Supervisor.
            project_id: GCP project id.
            location: GCP location (e.g. "europe-west1", "us-central1").
            system_prompt: System-level instruction → mapped to `system_instruction`.
            safety_settings: Optional list of Vertex safetySettings dicts (category + threshold).
                Defaults to None (no safety settings). For moderation use cases, use VertexModerationSupervisor.
            pre_processing: PreProcessing steps.
            api_key: OAuth2 bearer token (if given, overrides env).
            api_variable: Env var name for the token (e.g. "VERTEX_ACCESS_TOKEN").
        """
        self.name: str = model
        self.provider_name: str | None = "GoogleVertexAI"
        self.project_id = project_id
        self.location = location

        # Vertex generateContent endpoint
        self.base_url: str = (
            f"https://{location}-aiplatform.googleapis.com/v1/"
            f"projects/{project_id}/locations/{location}/publishers/google/"
            f"models/{model}:generateContent"
        )

        self.usage: Usage = usage
        self.res_map_fn: ResultMapper = cast(ResultMapper, partial(result_mapper, usage=self.usage))
        self.req_map_fn: RequestMapper[Self] = vertex_request_map
        self.auth_map_fn: AuthMapper = auth_map  # still Bearer <token>
        self.pre_processing = pre_processing
        self.api_key = api_key
        self.api_variable = api_variable

        # Vertex-specific settings
        self.system_prompt = system_prompt
        self.safety_settings = safety_settings or []

        super().__post_init__()

