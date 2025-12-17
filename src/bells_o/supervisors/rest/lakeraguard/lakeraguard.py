"""Implement the LakeraGuard supervisor via REST API."""

from typing import Self

from bells_o.common import AuthMapper, RequestMapper, ResultMapper, Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import lakeraguard as lakera_result_map
from bells_o.supervisors.rest.auth_mappers import auth_bearer as auth_map
from bells_o.supervisors.rest.request_mappers import lakeraguard as lakera_request_map

from ..rest_supervisor import RestSupervisor


class LakeraGuardSupervisor(RestSupervisor):
    """Implement the LakeraGuard supervisor via REST API, with any policy."""

    def __init__(
        self,
        project_id: str,
        usage: Usage,
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str | None = "LAKERA_API_KEY",
    ):
        """Initialize the custom LakeraGuardSupervisor with a custom policy.

        This implementation needs a specified `usage_type`. It assumes that the project of specified `project_id` can have any custom policy.

        Args:
            usage (Usage): The usage of the supervisor, defined by the policy of the passed `project_id`
            project_id (str): The id of the project to authenticate with.
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key (str | None, optional): API key to use, takes priority over `api_variable`. Defaults to None.
            api_variable (str | None, optional): Environment variable name that stores the API key. Defaults to "LAKERA_API_KEY".

        """
        self.name: str = "LakeraGuard"
        self.provider_name: str | None = "Lakera"
        self.base_url: str = "https://api.lakera.ai/v2/guard"
        self.usage: Usage = usage
        self.res_map_fn: ResultMapper = lakera_result_map
        self.req_map_fn: RequestMapper[Self] = lakera_request_map
        self.auth_map_fn: AuthMapper = auth_map
        self.pre_processing = pre_processing
        self.project_id = project_id
        self.api_key = api_key
        self.api_variable = api_variable
        self.custom_header = {}

        super().__post_init__()
