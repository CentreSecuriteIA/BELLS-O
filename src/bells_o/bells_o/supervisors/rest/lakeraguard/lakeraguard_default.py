"""Implement the LakeraGuard supervisor via REST API."""

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing

from ..lakeraguard import LakeraGuardSupervisor


class LakeraGuardDefaultSupervisor(LakeraGuardSupervisor):
    """Implement the LakeraGuard supervisor via REST API, with the default policy."""

    def __init__(
        self,
        project_id: str,
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str | None = None,
    ):
        """Initialize the LakeraGuardDefaultSupervisor.

        It assumes that the project of specified `project_id` has the default policy.

        Args:
            project_id (str): The id of the project to authenticate with.
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key (str | None, optional): API key to use, takes priority over `api_variable`. Defaults to None.
            api_variable (str | None, optional): Environment variable name that stores the API key. Defaults to "LAKERA_API_KEY"

        """
        super().__init__(
            usage=Usage("content_moderation", "prompt_injection"),
            project_id=project_id,
            pre_processing=pre_processing,
            api_key=api_key,
            api_variable=api_variable,
        )
        self.name: str = "LakeraGuard Default"
