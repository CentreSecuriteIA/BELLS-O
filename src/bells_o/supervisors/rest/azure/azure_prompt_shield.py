"""Implement the Analyze Text supervisor from Azure via REST API."""

from typing import Any

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import azure_prompt_shield as result_map
from bells_o.supervisors.rest.auth_mappers import ocp_apim_subscription as auth_map
from bells_o.supervisors.rest.request_mappers import azure_prompt_shield as request_map

from ..rest_supervisor import RestSupervisor


# TODO: handle error return, fails script rn because the reponse body is different
class AzurePromptShieldSupervisor(RestSupervisor):
    """Implement the Analyze Text supervisor from Azure via REST API with `api-version=2024-09-01`."""

    def __init__(
        self,
        endpoint: str,
        pre_processing: list[PreProcessing] = [],
        api_key: str | None = None,
        api_variable: str = "AZURE_API_KEY",
    ):
        """Initialize the AzureAnalyzeTextSupervisor with adjustable categories.

        This implementation needs a specified `usage_type`. It assumes that the project of specified `project_id` can have any custom policy.

        Args:
            endpoint (str): The endpoint URL to use. Requires an Azure resource for "AI Content Safet - Text Analyze" to be set up.
            categories (list[Literal["Hate", "Sexual", "SelfHarm", "Violence"]]): List of categories to analyze the text for. Defaults to ["Hate", "Sexual", "SelfHarm", "Violence"].
            severity_threshold (int): The severity threshold to classify a prompt as harmful. Defaults to 6.
            output_type (Literal["FourSeverityLevels", "EightSeverityLevels"]): The `outputType` parameter in the API request. See https://learn.microsoft.com/en-us/azure/ai-services/content-safety/quickstart-text?pivots=programming-language-rest.
            pre_processing (list[PreProcessing], optional): List of PreProcessing steps to apply to prompts. Defaults to [].
            api_key (str | None, optional): API key to use, takes priority over `api_variable`. Defaults to None.
            api_variable (str | None, optional): Environment variable name that stores the API key. Defaults to "AZURE_API_KEY"

        """
        super().__init__(
            name="Analyze text",
            usage=Usage("jailbreak"),
            res_map_fn=result_map,
            base_url=f"{endpoint}/contentsafety/text:shieldPrompt?api-version=2024-09-01",
            req_map_fn=request_map,
            auth_map_fn=auth_map,
            pre_processing=pre_processing,
            provider_name="Azure",
            api_key=api_key,
            api_variable=api_variable,
            custom_header={"Content-Type": "application/json"},
        )

    @classmethod
    def _get_token_counts(cls, output_raw: dict[str, Any]) -> dict[str, Any]:
        return {"input_tokens": 0, "output_tokens": 0}
