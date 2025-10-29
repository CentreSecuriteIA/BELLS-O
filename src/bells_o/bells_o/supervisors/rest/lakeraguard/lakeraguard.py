"""Implement the pre-configured saillab/x-guard supervisor from HuggingFace."""

from functools import partial

from bells_o.common import JsonMapper, ResultMapper, Usage
from bells_o.preprocessors import PreProcessing
from bells_o.resultmappers import lakeraguard as lakera_result_map
from bells_o.supervisors.rest.auth_mappers import auth_bearer as auth_map
from bells_o.supervisors.rest.requestmappers import lakeraguard as lakera_request_map

from ..custom_endpoint import RestSupervisor


# Usage("content_moderation", "prompt_injection")


class LakeraGuardSupervisor(RestSupervisor):
    """Implement the LakeraGuard supervisor via REST API, with any policy."""

    def __init__(
        self,
        project_id: str,
        usage: Usage,
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
        self.name: str = "LakeraGuard"
        self.provider_name: str | None = "Lakera"
        self.base_url: str = "https://api.lakera.ai/v2/guard"
        self.usage: Usage = usage
        self.res_map_fn: ResultMapper = partial(lakera_result_map, usage=self.usage)
        self.req_map_fn: JsonMapper = lakera_request_map
        self.auth_map_fn: JsonMapper = auth_map
        self.pre_processing = pre_processing
        self.project_id = project_id
        self.api_key = api_key
        self.api_variable = api_variable

        super().__post_init__()
