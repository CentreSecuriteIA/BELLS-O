"""Implement the OpenCC (Constitutional Classifier) supervisor via its local REST API."""

from typing import Any

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import opencc as opencc_result_map
from bells_o.supervisors.rest.auth_mappers import no_auth as auth_map
from bells_o.supervisors.rest.request_mappers import opencc as opencc_request_map

from ..rest_supervisor import RestSupervisor


class OpenCCSupervisor(RestSupervisor):
    """Implement the OpenCC constitutional-classifier pipeline via its local REST API.

    OpenCC is served locally (FastAPI) and exposes an unauthenticated ``POST /check`` endpoint.
    The pipeline returns a single terminal ``decision`` (``"allow"``, ``"block"`` or
    ``"annotate"``); anything other than ``"allow"`` is treated as a detection.
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8000,
        mode: str | None = None,
        pre_processing: list[PreProcessing] = [],
    ):
        """Initialize the OpenCCSupervisor.

        Args:
            host (str): Host the OpenCC server is bound to. Defaults to "127.0.0.1".
            port (int): Port the OpenCC server is listening on. Defaults to 8000.
            mode (str | None, optional): Override the server's terminal-flag action per request,
                either "block" or "annotate". Defaults to None, which uses the server's configured
                `default_action`.
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to
                prompts. Defaults to [].

        """
        self.mode = mode

        super().__init__(
            name="OpenCC",
            usage=Usage("jailbreak", "content_moderation"),
            res_map_fn=opencc_result_map,
            base_url=f"http://{host}:{port}/check",
            req_map_fn=opencc_request_map,
            auth_map_fn=auth_map,
            pre_processing=pre_processing,
            provider_name="OpenCC",
            needs_api=False,
        )

    @classmethod
    def _get_token_counts(cls, output_raw: dict[str, Any]) -> dict[str, Any]:
        return {"input_tokens": 0, "output_tokens": 0}
