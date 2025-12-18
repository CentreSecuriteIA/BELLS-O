"""Implement the LakeraGuard supervisor via REST API."""

from bells_o.common import Usage
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
        self.project_id = project_id

        super().__init__(
            name="LakeraGuard",
            usage=usage,
            res_map_fn=lakera_result_map,
            base_url="https://api.lakera.ai/v2/guard",
            req_map_fn=lakera_request_map,
            auth_map_fn=auth_map,
            pre_processing=pre_processing,
            provider_name="Lakera",
            api_key=api_key,
            api_variable=api_variable,
        )
