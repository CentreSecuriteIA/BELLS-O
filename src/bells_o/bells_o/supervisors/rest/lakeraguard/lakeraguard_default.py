"""Implement the pre-configured saillab/x-guard supervisor from HuggingFace."""

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing

from ..lakeraguard import LakeraGuardSupervisor


class LakeraGuardDefaultSupervisor(LakeraGuardSupervisor):
    """Implement the LakeraGuard supervisor via REST API, with any policy."""

    def __init__(
        self,
        project_id: str,
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str | None = None,
    ):
        """Initialize the custom LakeraGuardSupervisor with a custom policy.

        This implementation needs a specified `usage_type`. It assumes that the project of specified `project_id` can have any custom policy.

        Args:
            usage (Usage): _description_
            base_url (str): _description_
            project_id (str): _description_
            pre_processing (list[PreProcessing], optional): _description_. Defaults to [].
            api_key (str | None, optional): _description_. Defaults to None.
            api_variable (str | None, optional): _description_. Defaults to None.

        """
        super().__init__(
            usage=Usage("content_moderation", "prompt_injection"),
            project_id=project_id,
            pre_processing=pre_processing,
            api_key=api_key,
            api_variable=api_variable,
        )
        self.name: str = "LakeraGuard Default"
