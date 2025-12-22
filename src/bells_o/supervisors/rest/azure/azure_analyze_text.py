"""Implement the Analyze Text supervisor from Azure via REST API."""

from typing import Literal

from bells_o.common import Usage
from bells_o.preprocessors import PreProcessing
from bells_o.result_mappers import azure_analyze_text as azure_result_map
from bells_o.supervisors.rest.auth_mappers import ocp_apim_subscription as auth_map
from bells_o.supervisors.rest.request_mappers import azure_analyze_text as azure_request_map

from ..rest_supervisor import RestSupervisor


class AzureAnalyzeTextSupervisor(RestSupervisor):
    """Implement the Analyze Text supervisor from Azure via REST API with `api-version=2024-09-01`."""

    def __init__(
        self,
        endpoint: str,
        categories: list[Literal["Hate", "Sexual", "SelfHarm", "Violence"]] = [
            "Hate",
            "Sexual",
            "SelfHarm",
            "Violence",
        ],
        severity_threshold: int = 6,
        output_type: Literal["FourSeverityLevels", "EightSeverityLevels"] = "FourSeverityLevels",
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
        # TODO: think about adding severity thresholds in the usage object.
        self.categories = categories
        self.output_type = output_type
        super().__init__(
            name="Analyze text",
            usage=Usage("content_moderation"),
            res_map_fn=azure_result_map,
            base_url="{endpoint}/contentsafety/text:analyze?api-version=2024-09-01",
            req_map_fn=azure_request_map,
            auth_map_fn=auth_map,
            pre_processing=pre_processing,
            provider_name="Azure",
            api_key=api_key,
            api_variable=api_variable,
            custom_header={"Content-Type": "application/json"},
        )
